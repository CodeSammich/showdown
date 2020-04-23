import constants
from data import all_move_json
from showdown.battle import Battle
from showdown.engine.damage_calculator import calculate_damage
from showdown.engine.find_state_instructions import update_attacking_move
from ..helpers import format_decision

from showdown.battle_bots.nn_bot.deep_q_network import DeepQNetwork
from showdown.engine.evaluate import evaluate, evaluate2
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

    async def find_best_move(self, agent=None): # calls best_move to start even when it does not go first?
        state = self.create_state()
        my_options = self.get_all_options()[0] # all valid actions, already accounts for struggle and switches

        # all switch options, even if fainted or self
        all_switches = []
        for pkmn in self.all_pokemon:
            all_switches.append("{} {}".format(constants.SWITCH_STRING, pkmn.name))

        # Get all moves and switches, not being used right now
        # e.g. volt switch
        moves = []
        switches = []
        for option in my_options:
            if option.startswith(constants.SWITCH_STRING + " "):
                switches.append(option)
            else:
                moves.append(option)

        if self.force_switch or not moves:
            return format_decision(self, switches[0])

        # convert state to matrix
        matrix = self.state_to_vector()
        totalEnemyHealth = evaluate2(state)
        # Calculate New Reward
        if agent.previous_state is not None:

            await agent.step(agent.previous_state, agent.previous_action, (agent.previous_reward - totalEnemyHealth)/6, matrix, False)

        # pass through network and return choice
        idx, choice = agent.act(matrix, my_options, all_switches)
        agent.set_previous(matrix, idx, totalEnemyHealth)

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
        opp_act_vec = self.opponent.active.to_vector(False)
        state_matrix.append(opp_act_vec)

        # convert each of opponent's reserves to vectors
        reserve = self.opponent.reserve
        for pokemon in reserve:
            state_matrix.append(pokemon.to_vector(False))
 
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

        # Encode who is currently Active as 1-hot
        one_hot_pokemon = torch.IntTensor([int(self.user.active == poke) for poke in self.all_pokemon])
        state_matrix.append(one_hot_pokemon)

        # convert user's active to one vector
        state_matrix.append(self.user.active.to_vector())

        # convert each of user's reserves to vectors
        reserve = self.user.reserve
        for pokemon in reserve:
            state_matrix.append(pokemon.to_vector())

        # flatten final self matrix into one vector
        return torch.cat(state_matrix, dim=0)

