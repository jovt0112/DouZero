from copy import deepcopy
import random

from . import move_detector as md, move_selector as ms
from .move_generator import MovesGener

# 新增癞子映射（假设L1=21，L2=22）
EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D', 21: 'L1', 22: 'L2'}
RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30, 'L1': 21, 'L2': 22}

class GameEnv(object):
    def __init__(self, players):
        self.card_play_action_seq = []
        self.three_landlord_cards = None
        self.game_over = False
        self.acting_player_position = None
        self.player_utility_dict = None
        self.players = players
        self.last_move_dict = {'landlord': [], 
                               'landlord_up': [], 
                               'landlord_down': []}
        self.played_cards = {'landlord': [], 
                             'landlord_up': [], 
                             'landlord_down': []}
        self.last_move = []
        self.last_two_moves = []
        self.num_wins = {'landlord': 0, 
                         'farmer': 0}
        self.num_scores = {'landlord': 0, 
                           'farmer': 0}
        self.info_sets = {'landlord': InfoSet('landlord'), 
                          'landlord_up': InfoSet('landlord_up'), 
                          'landlord_down': InfoSet('landlord_down')}
        self.bomb_num = 0
        self.last_pid = 'landlord'
        self.tian_laizi = None  # 天癞子
        self.di_laizi = None    # 地癞子

    def card_play_init(self, card_play_data):
        # 初始化天癞子（随机抽取非大小王牌）
        base_cards = [3,4,5,6,7,8,9,10,11,12,13,14,17]
        self.tian_laizi = random.choice(base_cards)
        # 初始化地癞子（抢地主后抽取，此处简化为直接生成）
        self.di_laizi = random.choice([c for c in base_cards if c != self.tian_laizi])

        self.info_sets['landlord'].player_hand_cards = card_play_data['landlord']
        self.info_sets['landlord_up'].player_hand_cards = card_play_data['landlord_up']
        self.info_sets['landlord_down'].player_hand_cards = card_play_data['landlord_down']
        self.three_landlord_cards = card_play_data['three_landlord_cards']
        self.get_acting_player_position()
        self.game_infoset = self.get_infoset()

    def compute_player_utility(self):
        if len(self.info_sets['landlord'].player_hand_cards) == 0:
            self.player_utility_dict = {'landlord': 2, 
                                        'farmer': -1}
        else:
            self.player_utility_dict = {'landlord': -2, 
                                        'farmer': 1}

    def update_num_wins_scores(self):
        for pos, utility in self.player_utility_dict.items():
            base_score = 2 if pos == 'landlord' else 1
            if utility > 0:
                self.num_wins[pos] += 1
                self.winner = pos
                # 计算炸弹倍数
                multiplier = 1
                last_action = self.card_play_action_seq[-1] if self.card_play_action_seq else []
                if md.is_soft_bomb(last_action, self.tian_laizi, self.di_laizi):
                    multiplier = 2
                elif md.is_hard_bomb(last_action) or md.is_pure_laizi_bomb(last_action):
                    multiplier = 4
                elif (md.is_laizi_bomb(last_action, self.tian_laizi, self.di_laizi) 
                      and len(last_action) >=5) or md.is_wangzha(last_action):
                    multiplier = 6
                self.num_scores[pos] += base_score * (2 ** self.bomb_num) * multiplier
            else:
                self.num_scores[pos] -= base_score * (2 ** self.bomb_num)

    def step(self):
        action = self.players[self.acting_player_position].act(self.game_infoset)
        assert action in self.game_infoset.legal_actions
        
        # 处理炸弹类型
        if md.is_bomb(action, self.tian_laizi, self.di_laizi):
            self.bomb_num += 1
        
        self.last_move_dict[self.acting_player_position] = action.copy()
        self.card_play_action_seq.append(action)
        self.update_acting_player_hand_cards(action)
        self.played_cards[self.acting_player_position] += action
        
        if self.acting_player_position == 'landlord' and len(action) > 0:
            for card in action:
                if card in self.three_landlord_cards:
                    self.three_landlord_cards.remove(card)
        
        self.game_done()
        if not self.game_over:
            self.get_acting_player_position()
            self.game_infoset = self.get_infoset()

    def get_legal_card_play_actions(self):
        mg = MovesGener(self.info_sets[self.acting_player_position].player_hand_cards,
                        self.tian_laizi, self.di_laizi)
        action_sequence = self.card_play_action_seq
        rival_move = action_sequence[-1] if action_sequence else []
        
        rival_type = md.get_move_type(rival_move, self.tian_laizi, self.di_laizi)
        rival_move_type = rival_type['type']
        rival_move_len = rival_type.get('len', 1)
        
        moves = []
        if rival_move_type == md.TYPE_0_PASS:
            moves = mg.gen_all_moves()
        else:
            moves = mg.gen_valid_moves(rival_move_type, rival_move_len)
        
        # 添加炸弹选项
        if rival_move_type not in [md.TYPE_0_PASS, md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB]:
            moves += mg.gen_bombs()
        
        # 添加过牌选项
        if len(rival_move) != 0:
            moves.append([])
        
        return moves

    def get_infoset(self):
        info = deepcopy(self.info_sets[self.acting_player_position])
        info.legal_actions = self.get_legal_card_play_actions()
        info.bomb_num = self.bomb_num
        info.last_move = self.get_last_move()
        info.last_two_moves = self.get_last_two_moves()
        info.last_move_dict = self.last_move_dict
        info.num_cards_left_dict = {
            pos: len(self.info_sets[pos].player_hand_cards) 
            for pos in ['landlord', 'landlord_up', 'landlord_down']
        }
        info.other_hand_cards = [
            c for pos in ['landlord', 'landlord_up', 'landlord_down'] 
            if pos != self.acting_player_position 
            for c in self.info_sets[pos].player_hand_cards
        ]
        info.played_cards = self.played_cards
        info.three_landlord_cards = self.three_landlord_cards
        info.card_play_action_seq = self.card_play_action_seq
        info.all_handcards = {
            pos: self.info_sets[pos].player_hand_cards 
            for pos in ['landlord', 'landlord_up', 'landlord_down']
        }
        info.last_pid = self.last_pid
        info.tian_laizi = self.tian_laizi  # 新增癞子信息
        info.di_laizi = self.di_laizi
        return info

class InfoSet(object):
    def __init__(self, player_position):
        self.player_position = player_position
        self.player_hand_cards = None
        self.num_cards_left_dict = None
        self.three_landlord_cards = None
        self.card_play_action_seq = None
        self.other_hand_cards = None
        self.legal_actions = None
        self.last_move = None
        self.last_two_moves = None
        self.last_move_dict = None
        self.played_cards = None
        self.all_handcards = None
        self.last_pid = None
        self.bomb_num = 0
        self.tian_laizi = None  # 新增癞子信息
        self.di_laizi = None
