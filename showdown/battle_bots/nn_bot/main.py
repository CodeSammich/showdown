import constants
from data import all_move_json
from showdown.battle import Battle
from showdown.engine.damage_calculator import calculate_damage
from showdown.engine.find_state_instructions import update_attacking_move
from ..helpers import format_decision

from showdown.battle_bots.nn_bot.deep_q_network import DeepQNetwork
from showdown.engine.evaluate import evaluate
import torch
import torch.nn as nn
import torch.optim as optim

class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)
        # save model after init, can't make object attribute due to pickling errors w/ deepcopy
        #torch.save(model.state_dict(), 'nn_bot_trained')

        # init env conditions for vectorize
        self.weather_conds = [constants.RAIN, 
                              constants.SUN,
                              constants.SAND,
                              constants.HAIL]
        self.field_conds = [constants.ELECTRIC_TERRAIN,
                           constants.GRASSY_TERRAIN,
                           constants.MISTY_TERRAIN,
                           constants.PSYCHIC_TERRAIN]
        self.side_conds = list(constants.COURT_CHANGE_SWAPS) # init for stable
        self.previous_state = None
        self.previous_action = None

    def find_best_move(self, agent=None): # calls best_move to start even when it does not go first?
        # network = DeepQNetwork()
        # criterion = nn.MSELoss()
        # optimizer = optim.Adam(network.parameters())
        state = self.create_state()
        my_options = self.get_all_options()[0]
        # Get all options (even impossible)
        # all_moves = self.user.active.moves
        # for pkmn in self.user.reserve:
        #     all_moves.append("{} {}".format(constants.SWITCH_STRING, pkmn.name))
        # print(all_moves)
        # l = []
        # for item in all_moves:
        #     l.append(int(item in my_options))
        # print(l)

        moves = []
        switches = []
       # print('possible moves: {}'.format(my_options))
        for option in my_options:
            if option.startswith(constants.SWITCH_STRING + " "):
                switches.append(option)
            else:
                moves.append(option)

        if self.force_switch or not moves:
            return format_decision(self, switches[0])

        # THIS WHOLE SECTION SHOULD BE REPLACED BY "find_move" in Agent class
        # pick moves logic for non-nn based on simple most damage formula
        if agent == None:
            most_damage = -1
            choice = None
            for move in moves:
                # pick best move 
                # most damage per move
                damage_amounts = calculate_damage(state, constants.SELF, move, constants.DO_NOTHING_MOVE)

                damage = damage_amounts[0] if damage_amounts else 0

                if damage > most_damage:
                    choice = move
                    most_damage = damage
        else:
            ##### TODO: wrap through Agent later on, this is just for testing
            # convert state to matrix via state_to_matrix
            # Calculate New Reward
            if self.previous_state:
                agent.step(self.previous_state, self.previous_action, evaluate(state), state)

            matrix = self.state_to_vector()
            # reinitialize and load weights
            # model = DeepQNetwork() 
            # model.load_state_dict(torch.load('nn_bot_trained'))


            # expected utility Q(s,a) = R_{t+1} + gamma*max_a{[Q(s,a)]}
            # loss: output vs. expected utility 

            # pass input thorugh
            ind = agent.act(matrix)
            # matrix = model(matrix.float())
            # print('logits layer: matrix', matrix)
            # TODO: account for my_options shrinking due to death
            # ind = matrix[0:len(my_options)].argmax()
            ind = ind % len(my_options)
            self.previous_state = state
            self.previous_action = ind

            # get logits layer and take the best moves (highest value after softmax)
            choice = my_options[ind]

        return format_decision(self, choice)

    def state_to_vector(self):
        '''
        converts current battle state into a 1D vector

        [weather, field, trick_room, 
         opp_wish, opp_side_cond, opp_active, 
         user_wish, user_side_cond, user_active, user_reserve]

        drawback: it only sees opponent's current pokemon
        note: user_reserve is at end due to variability

        output: torch.intTensor (1D)
        '''
        state_matrix = []

        ####### ENVIRONMENT #######
        # convert weather to a 6x1 tensor of 0s, set approriate weather flag to 1
        one_hot_weather = [int(w == self.weather) for w in self.weather_conds] 
        weather_vec = torch.IntTensor(one_hot_weather)
        state_matrix.append(weather_vec)

        # convert field to a 4x1 zeros tensor, set appropriate flag to 1
        one_hot_field = [int(w == self.field) for w in self.field_conds] #LET MY PEOPLE GO
        field_vec = torch.IntTensor(one_hot_field)
        state_matrix.append(field_vec)

        # convert field to a 1x1  tensor
        trick_room_vec = torch.IntTensor([self.trick_room])
        state_matrix.append(trick_room_vec)

        ###### OPPONENT ########
        # convert opponent's wish to one vector of 1 or 0
        opp_wish = 1 if self.opponent.wish[0] > 0 else 0
        opp_wish_vec = torch.IntTensor([opp_wish])
        state_matrix.append(opp_wish_vec)

        # convert opponent's side conditions to one vector
        # Spikes range from 0 to 3, Toxic Spikes range from 0 to 2
        # for all side condition keys, put corresponding value in stable order
        opp_side_cond_dict = self.opponent.side_conditions # dict (str -> int)
        opp_side_cond = [opp_side_cond_dict[cond] for cond in self.side_conds] 
        opp_side_cond_vec = torch.IntTensor(opp_side_cond)
        state_matrix.append(opp_side_cond_vec)
        
        # convert opponent's active to one vector
        opp_act_vec = self.opponent.active.to_vector()
        state_matrix.append(opp_act_vec)
 
        ######## USER #########
        # convert user's wish to one vector
        user_wish = 1 if self.user.wish[0] > 0 else 0
        user_wish_vec = torch.IntTensor([user_wish])
        state_matrix.append(user_wish_vec)

        # convert user's side conditions to one vector
        # Spikes range from 0 to 3, Toxic Spikes range from 0 to 2
        # for all side condition keys, put corresponding value in stable order
        user_side_cond_dict = self.user.side_conditions # dict (str -> int)
        user_side_cond = [user_side_cond_dict[cond] for cond in self.side_conds] 
        user_side_cond_vec = torch.IntTensor(user_side_cond)
        state_matrix.append(user_side_cond_vec)

        # convert user's active to one vector
        state_matrix.append(self.user.active.to_vector())

        # convert each of user's reserves to vectors
        user_reserve = self.user.reserve
        for pokemon in user_reserve:
            state_matrix.append(pokemon.to_vector())

        # flatten final self matrix into one vector
        return torch.cat(state_matrix, dim=0)

