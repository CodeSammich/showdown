import asyncio
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

from data import all_move_json
from data import pokedex
from data.mods.apply_mods import apply_mods
from showdown.battle_bots.nn_bot.DQNAgent import Agent

logger = logging.getLogger(__name__)

def create_challenge_bot():
    conf = Config()
    conf.battle_bot_module = "nn_bot"
    conf.save_replay = config.save_replay
    conf.use_relative_weights = config.use_relative_weights
    conf.gambit_exe_path = config.gambit_exe_path
    conf.search_depth = config.search_depth
    conf.websocket_uri = "localhost:8000"
    conf.username = "cbninjask5uber"
    conf.password = "aiaccount1"
    conf.bot_mode = "CHALLENGE_USER"
    conf.team_name = "gen8/ou/clef_sand"
    conf.pokemon_mode = "gen8ou"
    conf.run_count = 1
    conf.user_to_challenge = "AcceptGary"
    conf.LOG_LEVEL = 'DEBUG'
    return conf

def create_accept_bot():
    """
    A hardcoded bot
    """
    conf = Config()
    conf.battle_bot_module = "rand_bot"
    conf.save_replay = config.save_replay
    conf.use_relative_weights = config.use_relative_weights
    conf.gambit_exe_path = config.gambit_exe_path
    conf.search_depth = config.search_depth
    conf.websocket_uri = "localhost:8000"
    conf.username = "AcceptGary"
    conf.password = "password"
    conf.bot_mode = "ACCEPT_CHALLENGE"
    conf.team_name = "gen8/ou/band_toad"
    conf.pokemon_mode = "gen8ou"
    conf.run_count = 1
    conf.LOG_LEVEL = 'DEBUG'
    return conf

def force_global_config(conf):
    config.pokemon_mode = conf.pokemon_mode
    config.greeting_message = conf.greeting_message
    config.gambit_exe_path = conf.gambit_exe_path

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


async def showdown(agent = None):
    if agent == None:
        conf = create_accept_bot()  # hardcoded agent
    else:
        conf = create_challenge_bot()  # nn bot

    config = conf
    init_logging("DEBUG")
    apply_mods(config.pokemon_mode)

    original_pokedex = deepcopy(pokedex)
    original_move_json = deepcopy(all_move_json)

    ps_websocket_client = await PSWebsocketClient.create(config.username, config.password, config.websocket_uri)
    await ps_websocket_client.login()

    battles_run = 0
    wins = 0
    losses = 0
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

        winner = await pokemon_battle(ps_websocket_client, config.pokemon_mode, config, agent)

        if winner == config.username:
            wins += 1
        else:
            losses += 1

        logger.info("W: {}\tL: {}".format(wins, losses))

        check_dictionaries_are_unmodified(original_pokedex, original_move_json)

        battles_run += 1
        if battles_run >= config.run_count:
            break

async def main():
    """
    Goal of function is to put a neural network bot versus a hard coded bot
    """
    agent = Agent(8175, 10, 42)
    await asyncio.gather(showdown(agent), showdown())
    #agent = Agent(8175, 10, 42)
    #agent2 = Agent(8175, 10, 42)
    #agent3 = Agent(8175, 10, 42)
    #await asyncio.gather(showdown(agent), showdown(), ..., ...) # specify one of the agents as being a neural network
    #combine_weights(agent, agent2, agent3)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
    print("done")