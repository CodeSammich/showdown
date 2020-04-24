import asyncio
import numpy as np
import json
from copy import deepcopy
from  async_timeout import timeout
import matplotlib.pyplot as plt

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
from showdown.battle_bots.nn_bot.deep_q_network import DeepQNetwork
import time
import torch
from showdown.engine.evaluate import evaluate

logger = logging.getLogger(__name__)

# Constants
ENEMY_BOT = "most_damage"
ENEMY_TEAM = "random"
POSSIBLE_TEAMS = ["clef_sand", "band_toad", "balance", "simple", "weavile_stall", "mew_stall"]
LOG_MODE = "CRITICAL"
LOAD = False
SAVE = True
TRAIN = True
"""Training params"""
num_games = 3
TIMEOUT = 200 #seconds
<<<<<<< HEAD
episodes = 300 
=======
episodes = 200
>>>>>>> f65602cc2ffafcf23607d417b9c75e71096f285f
merge_networks_time = 10000  # run this many times and then merge multiple agents TODO

"""Performance Params"""
eval_time = 5  # evals the network every eval_time steps
eval_run_battles = 1  # runs this many battles to determine performance against
# eval_opponent = "safest"  # what is the neural network evaluating against
randRewardList = []
damageRewardList = []
episodeList = []
total_wins_rand = []
total_wins_damage = []

seed = np.random.randint(0, 50)

"""Network Params"""
state_size = 405
actions = 25 

def create_challenge_bot(one = True, bot = ENEMY_BOT):
    """
    agent: specify str agent name (i.e rand_bot) or a DQNAgent. If DQNAgent will choose nn_bot
    """
    conf = Config()
    conf.battle_bot_module = bot
    conf.save_replay = config.save_replay
    conf.use_relative_weights = config.use_relative_weights
    conf.gambit_exe_path = config.gambit_exe_path
    conf.search_depth = config.search_depth
    conf.websocket_uri = "localhost:8000"
    if one:
        conf.username = "cbninjask5uber"
        conf.password = "aiaccount1"
    else:
        conf.username = "MonkeyAttak"
        conf.password = "424242"
    conf.bot_mode = "CHALLENGE_USER"
    conf.team_name = ENEMY_TEAM
    conf.pokemon_mode = "gen8ou"
    conf.run_count = 1
    if one:
        conf.user_to_challenge = "AcceptGary"
    else:
        conf.user_to_challenge = "AcceptGary2"
    conf.LOG_LEVEL = 'DEBUG'
    return conf

def create_accept_bot(one=True):
    """
    agent: specify str agent name (i.e rand_bot) or a DQNAgent. If DQNAgent will choose nn_bot
    """
    conf = Config()
    conf.battle_bot_module = "nn_bot"
    conf.save_replay = config.save_replay
    conf.use_relative_weights = config.use_relative_weights
    conf.gambit_exe_path = config.gambit_exe_path
    conf.search_depth = config.search_depth
    conf.websocket_uri = "localhost:8000"
    if one:
        conf.username = "AcceptGary"
        conf.password = "password"
    else:
        conf.username = "AcceptGary2"
        conf.password = "password2"
    conf.bot_mode = "ACCEPT_CHALLENGE"
    conf.team_name = "gen8/ou/simple"
    conf.pokemon_mode = "gen8ou"
    conf.run_count = 1
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


async def showdown(accept, agent=None):
    """
    Will run through number of battles specified in config

    accept: boolean. If accept is true will create an accept_bot
    agent: specify str agent name (i.e rand_bot) or a DQNAgent. If DQNAgent will choose nn_bot
    """
    if accept:
        conf = agent.config
        # conf = create_accept_bot(agent)  # accept gary
    else:
        await asyncio.sleep(1)  # ensure that challenge bot has time to be created first
        if agent == "True" or agent == "rand_bot" or agent == "safest":
            if agent == "True": agent = ENEMY_BOT
            conf = create_challenge_bot(True, agent)  # cbninjask5uber
        else:
            if agent == "False": agent = ENEMY_BOT
            conf = create_challenge_bot(False, agent)  # cbninjask5uber

    config = conf
    apply_mods(config.pokemon_mode)

    original_pokedex = deepcopy(pokedex)
    original_move_json = deepcopy(all_move_json)

    ps_websocket_client = await PSWebsocketClient.create(config.username, config.password, config.websocket_uri)
    await ps_websocket_client.login()

    wins = 0
    losses = 0
    reward = 0

    for _ in range(num_games):
        if config.team_name != "random":
            team_name = config.team_name
        else:
            team_name = "gen8/ou/" + np.random.choice(POSSIBLE_TEAMS)
            # print(team_name)
        team = load_team(team_name)
        if config.bot_mode == constants.CHALLENGE_USER:
            await ps_websocket_client.challenge_user(config.user_to_challenge, config.pokemon_mode, team)
        elif config.bot_mode == constants.ACCEPT_CHALLENGE:
            await ps_websocket_client.accept_challenge(config.pokemon_mode, team)
        elif config.bot_mode == constants.SEARCH_LADDER:
            await ps_websocket_client.search_for_match(config.pokemon_mode, team)
        else:
            raise ValueError("Invalid Bot Mode")

        if type(agent) == str:
            winner = await pokemon_battle(ps_websocket_client, config.pokemon_mode, config, agent=None)
        else:
            winner = await pokemon_battle(ps_websocket_client, config.pokemon_mode, config, agent)

        if winner == config.username:
            finalReward = 1
            wins += 1
        else:
            finalReward = 0
            losses += 1

        if type(agent) != str:
            # logger.critical("W: {}\tL: {}".format(wins, losses))
            reward += agent.previous_reward + finalReward
            # logger.critical("End Score: {}".format(reward))
            # winPercList.append(reward)
            await agent.step(agent.previous_state, agent.previous_action, finalReward, agent.previous_state, True)
        else:
            logger.debug("W: {}\tL: {}".format(wins, losses))

    if type(agent) != str:
        reward = reward/num_games
        logger.critical("W: {}\tL: {}".format(wins, losses))
        logger.critical("End Score: {}".format(reward))
    check_dictionaries_are_unmodified(original_pokedex, original_move_json)
    return reward, wins #winner == config.username

    # battles_run += 1
    # if battles_run >= config.run_count:
    #     break

    # if accept:
    #     logger.critical("W: {}\tL: {}".format(wins, losses))
    #     logger.critical("End Score: {}".format(total_score/(wins + losses)))
    #
    # return
async def train_episode(agent1, agent2, agent3, agent4):
    """
    Goal of function is to put a neural network bot versus a hard coded bot
    """
    return await asyncio.gather(
        showdown(accept=True, agent=agent1), 
        showdown(accept=False, agent=agent2),
        showdown(accept=True, agent=agent3), 
        showdown(accept=False, agent=agent4))

async def main():
    """Call this code only once"""
    init_logging(LOG_MODE)

    agent1 = Agent(state_size, actions, seed, create_accept_bot(one = True))
    agent2 = Agent(state_size, actions, seed, create_accept_bot(one = False))
    agent1.lossList = agent2.lossList

    agent1.train()  # agent is in training mode
    agent2.train()
    # agent2 = "rand_bot" # two agents should actually be playing against eachother

    # reinitialize and load weights
    if LOAD:
        checkpoint = torch.load('nn_bot_trained')
        model = DeepQNetwork(state_size, actions)
        model.load_state_dict(checkpoint['local'])
        agent1.qnetwork_local = model
        model.load_state_dict(checkpoint['target'])
        agent1.qnetwork_target = model

    # Set networks to be equal
    agent2.qnetwork_local = agent1.qnetwork_local
    agent2.qnetwork_target = agent1.qnetwork_target
    if TRAIN:
        agent1.train()
        agent2.train()
    else:
        agent1.eval()
        agent2.eval()

    """main training loop"""
    lossList = []
    for episode in range(episodes):
        print("episode", episode)
        # await asyncio.sleep(30)
        start = time.time()
        try: # can probably run some of these in parallel using gather
            # await asyncio.wait_for(eternity(), timeout=1.0)
            await asyncio.wait_for(train_episode(agent1, "True", agent2, "False"), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            print('Timeout on iteration: ' + str(episode))
        print("Elapsed", time.time() - start)

        if (episode+1) % merge_networks_time == 0:
            pass  # TODO merge multiple agents

        if (episode + 1) % eval_time == 0:
            agent1.eval()
            agent2.eval()
            for _ in range(eval_run_battles):
                print("Eval Number", episode//eval_time)
                try: 
                    randReward, _, damageReward, _ = await asyncio.wait_for(
                        train_episode(agent1, "safest", agent2, "most_damage"), timeout=TIMEOUT)
                    episodeList.append(episode)
                    randRewardList.append(randReward[0]) # rewards
                    damageRewardList.append(damageReward[0]) # rewards
                    total_wins_rand.append(randReward[1]) # total wins
                    total_wins_damage.append(damageReward[1]) # total wins
                except asyncio.TimeoutError:
                    print('Timeout on test run: ' + str(episode//eval_time))
            agent1.train()  # allows the agent to train again
            agent2.train()  # allows the agent to train again

        # print dqn loss
        episodeLoss = np.mean(agent1.lossList)
        print(episodeLoss)
        lossList.append(episodeLoss)
        print(episodeLoss)
        agent1.lossList = []
        agent2.lossList = agent1.lossList

    if SAVE:
        torch.save({
            'local': agent1.qnetwork_local.state_dict(),
            'target': agent1.qnetwork_target.state_dict()
        }, 'nn_bot_trained')
        # Plot Graphs
    plt.clf()
    plt.figure()
    plt.plot(range(episode+1), lossList, label="agent loss list")
    plt.legend()
    plt.savefig("Agent_loss_421.png")
    plt.figure()
    plt.plot(episodeList, damageRewardList, label="Damage bot Curve")
    plt.plot(episodeList, randRewardList, label="Minimax bot Curve")
    plt.legend()
    plt.savefig("damage_minimax_421.png")
    plt.figure()
    plt.title("Total Wins per Episode")
    plt.plot(episodeList, total_wins_rand, label="Rand Bot")
    plt.plot(episodeList, total_wins_damage, label="Damage Bot")
    plt.legend()
    plt.show()
    print("done training")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
    print("done")

