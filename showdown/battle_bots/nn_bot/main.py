import constants
from data import all_move_json
from showdown.battle import Battle
from showdown.engine.damage_calculator import calculate_damage
from showdown.engine.find_state_instructions import update_attacking_move
from ..helpers import format_decision

from showdown.battle_bots.nn_bot.deep_q_network import DeepQNetwork
import torch.nn as nn
import torch.optim as optim

class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)

    def find_best_move(self): # calls best_move to start even when it does not go first?
        network = DeepQNetwork()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(network.parameters())
        state = self.create_state()
        my_options = self.get_all_options()[0]

        moves = []
        switches = []
        for option in my_options:
            if option.startswith(constants.SWITCH_STRING + " "):
                switches.append(option)
            else:
                moves.append(option)

        if self.force_switch or not moves:
            return format_decision(self, switches[0])

        # THIS WHOLE SECTION SHOULD BE REPLACED BY "find_move" in Agent class
        # pick moves logic for non-nn based on simple most damage formula
        if network == None:
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
            matrix = self.state_to_matrix(state)
            # flatten and feed that matrix to self.network
            
            # get logits layer and take the best moves (highest value after softmax)
            choice = move

        return format_decision(self, choice)

    def state_to_matrix(self, state):
        '''
        converts State into a vector
        [weather, field, trick_room, opp_active, user_active, user_reserve]
        drawback: it only sees opponent's current pokemon

        state: current State object from user's point of view
        output: torch.intTensor
        '''
        # ignoring wish and side_conditions(?) b/c it seems like noise for most situations
        # TODO: find out what side conditions is
        state_matrix = []

        # convert weather to a 6x1 tensor of 0s, set approriate weather flag to 1
        weather = [constants.RAIN, constants.SUN, constants.SAND, \
                constants.HAIL, constants.DESOLATE_LAND, constants.HEAVY_RAIN]
        one_hot_weather = [int(w == state.weather) for w in weather] # dang, it's hot out!
        weather_vec = torch.IntTensor(one_hot_weather)
        state_matrix.append(weather_vec)

        # convert field to a ?x? zeros tensor, set appropriate flag to 1
        # TODO: what can field be?
        field = state.field

        # convert field to a 1x1  tensor
        trick_room_vec = torch.IntTensor([state.trick_room])
        state_matrix.append(trick_room_vec)

        # convert opponent's active to one vector
        opp_act_vec = state.opponent.active.to_vector()
        state_matrix.append(opp_act_vec)

        # convert user's active to one vector
        user_act_vec = state.self.active.to_vector()
        state_matrix.append(user_act_vec)

        # convert each of user's reserves to vectors
        user_reserve = state.self.reserve
        for pokemon in user_reserve:
            state_matrix.append(pokemon.to_vector())

        # flatten final state matrix into one vector
        return torch.cat(state_matrix, dim=0)

