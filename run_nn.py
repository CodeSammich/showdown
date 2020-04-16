import asyncio
import numpy as np
import json
from copy import deepcopy

from environs import Env

import constants
import config
from config import init_logging, Config
import logging

from teams import load_team
#from showdown.run_battle import pokemon_battle
from showdown.run_train import pokemon_battle
from showdown.websocket_client import PSWebsocketClient

from  async_timeout import timeout
from data import all_move_json
from data import pokedex
from data.mods.apply_mods import apply_mods
from showdown.battle_bots.nn_bot.DQNAgent import Agent
from showdown.battle_bots.nn_bot.deep_q_network import DeepQNetwork
import time
import torch
from showdown.battle_bots.nn_bot.deep_q_network import DeepQNetwork
from showdown.engine.evaluate import evaluate

logger = logging.getLogger(__name__)

def create_challenge_bot(agent):
    """
    agent: specify str agent name (i.e rand_bot) or a DQNAgent. If DQNAgent will choose nn_bot
    """
    conf = Config()
    if type(agent) == type(""):
        conf.battle_bot_module = "most_damage"
    else:
        conf.battle_bot_module = "nn_bot"
    conf.save_replay = config.save_replay
    conf.use_relative_weights = config.use_relative_weights
    conf.gambit_exe_path = config.gambit_exe_path
    conf.search_depth = config.search_depth
    conf.websocket_uri = "localhost:8000"
    conf.username = "cbninjask5uber"
    conf.password = "aiaccount1"
    conf.bot_mode = "CHALLENGE_USER"
    conf.team_name = "gen8/ou/band_toad"
    conf.pokemon_mode = "gen8ou"
    conf.run_count = 5
    conf.user_to_challenge = "AcceptGary"
    conf.LOG_LEVEL = 'DEBUG'
    return conf

def create_accept_bot(agent):
    """
    agent: specify str agent name (i.e rand_bot) or a DQNAgent. If DQNAgent will choose nn_bot
    """
    conf = Config()
    if type(agent) == type(""):
        conf.battle_bot_module = "rand_bot"
    else:
        conf.battle_bot_module = "nn_bot"
    conf.save_replay = config.save_replay
    conf.use_relative_weights = config.use_relative_weights
    conf.gambit_exe_path = config.gambit_exe_path
    conf.search_depth = config.search_depth
    conf.websocket_uri = "localhost:8000"
    conf.username = "AcceptGary"
    conf.password = "password"
    conf.bot_mode = "ACCEPT_CHALLENGE"
    conf.team_name = "gen8/ou/band_toad"
    # conf.team_name = "gen8/ou/clef_sand"
    conf.pokemon_mode = "gen8ou"
    conf.run_count = 5
    conf.LOG_LEVEL = 'DEBUG'
    return conf

def check_dictionaries_are_unmodified(original_pokedex, original_move_json):
    # The bot should not modify the data dictionaries
    # This is a "just-in-case" check to make sure and will stop the bot if it mutates either of them
    if original_move_json != all_move_json:
        logger.critical("Move JSON changed!\nDumping modified version to `modified_moves.json`")
        with open("modified_moves.json", 'w') as f:
            json.dump(all_move_json, f, indent=4)
        exit(1)
    else:
        logger.debug("Move JSON unmodified!")

    if original_pokedex != pokedex:
        logger.critical("Pokedex JSON changed!\nDumping modified version to `modified_pokedex.json`")
        with open("modified_pokedex.json", 'w') as f:
            json.dump(pokedex, f, indent=4)
        exit(1)
    else:
        logger.debug("Pokedex JSON unmodified!")


async def showdown(accept, agent = None):
    """
    Will run through number of battles specified in config

    accept: boolean. If accept is true will create an accept_bot
    agent: specify str agent name (i.e rand_bot) or a DQNAgent. If DQNAgent will choose nn_bot
    """
    if accept:
        conf = create_accept_bot(agent)  # accept gary
    else:
        conf = create_challenge_bot(agent)  # cbninjask5uber

    config = conf
    apply_mods(config.pokemon_mode)

    original_pokedex = deepcopy(pokedex)
    original_move_json = deepcopy(all_move_json)

    ps_websocket_client = await PSWebsocketClient.create(config.username, config.password, config.websocket_uri)
    await ps_websocket_client.login()

    battles_run = 0
    wins = 0
    losses = 0
    total_score = 0
    while True:
        team = load_team(config.team_name)
        if config.bot_mode == constants.CHALLENGE_USER:
            await ps_websocket_client.challenge_user(config.user_to_challenge, config.pokemon_mode, team)
        elif config.bot_mode == constants.ACCEPT_CHALLENGE:
            await ps_websocket_client.accept_challenge(config.pokemon_mode, team)
        elif config.bot_mode == constants.SEARCH_LADDER:
            await ps_websocket_client.search_for_match(config.pokemon_mode, team)
        else:
            raise ValueError("Invalid Bot Mode")
        if type(agent) == type(""):
            winner = await pokemon_battle(ps_websocket_client, config.pokemon_mode, config, agent = None)
        else:
            winner = await pokemon_battle(ps_websocket_client, config.pokemon_mode, config, agent)

        if winner == config.username:
            finalReward = 1000
            wins += 1
        else:
            finalReward = -1000
            losses += 1
        
        if accept:
            logger.debug("W: {}\tL: {}".format(wins, losses))
            score = agent.memory.memory.copy().pop()[2]
            total_score += score + finalReward
            logger.debug("End Score: {}".format(score)) #
            agent.step(agent.previous_state, agent.previous_action, finalReward, torch.zeros(8175), True)
        else:
            logger.debug("W: {}\tL: {}".format(wins, losses))





        check_dictionaries_are_unmodified(original_pokedex, original_move_json)

        battles_run += 1
        if battles_run >= config.run_count:
            break

    if accept:
        logger.critical("W: {}\tL: {}".format(wins, losses))
        logger.critical("End Score: {}".format(total_score/(wins + losses)))
    

async def train_episode(agent1, agent2):
    """
    Goal of function is to put a neural network bot versus a hard coded bot
    """
    await asyncio.gather(showdown(accept=True, agent=agent1), showdown(accept=False, agent=agent2))

async def main():
    """Call this code only once"""
    init_logging("CRITICAL")

    """Training params"""
    episodes = 10
    state_size = 8175
    actions = 9
    merge_networks_time = 20  # run this many times and then merge multiple agents TODO
    seed = np.random.randint(0, 50)
    agent1 = Agent(state_size, actions, seed)

    # reinitialize and load weights
    checkpoint = torch.load('nn_bot_trained')
    model = DeepQNetwork(state_size, actions) 
    model.load_state_dict(checkpoint['local'])
    agent1.qnetwork_local = model
    model.load_state_dict(checkpoint['target'])
    agent1.qnetwork_target = model

    """main training loop"""
    for episode in range(episodes):
        print("episode", episode)
        agent2 = "rand_bot"
        try:
            async with timeout(10800) as cm:
                start = time.time()
                await train_episode(agent1, agent2=agent2) # can probably run some of these in parallel using gather
        except asyncio.TimeoutError as e:
            print("Elapsed", time.time() - start)
            print(e)
            print("Agent Timed Out! This is a problem")
        # logger.critical("End Score: {}".format(agent1.memory.memory.copy().pop()[2]))

        if (episode+1) % merge_networks_time == 0:
            pass  # TODO
    torch.save({
        'local': agent1.qnetwork_local.state_dict(),
        'target': agent1.qnetwork_target.state_dict()
    }, "nn_bot_trained")
    print("done training")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
    print("done")