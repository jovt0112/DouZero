import itertools

# global parameters
MIN_SINGLE_CARDS = 5
MIN_PAIRS = 3
MIN_TRIPLES = 2

# action types
TYPE_0_PASS = 0
TYPE_1_SINGLE = 1
TYPE_2_PAIR = 2
TYPE_3_TRIPLE = 3
TYPE_4_BOMB = 4
TYPE_5_KING_BOMB = 5
TYPE_6_3_1 = 6
TYPE_7_3_2 = 7
TYPE_8_SERIAL_SINGLE = 8
TYPE_9_SERIAL_PAIR = 9
TYPE_10_SERIAL_TRIPLE = 10
TYPE_11_SERIAL_3_1 = 11
TYPE_12_SERIAL_3_2 = 12
TYPE_13_4_2 = 13
TYPE_14_4_22 = 14
TYPE_15_WRONG = 15
TYPE_SOFT_BOMB = 16  # 软炸，由 1 - 4 张相同点数非癞子牌加上 1 - 8 张癞子牌组成的 ≥ 四张的牌型
TYPE_JOKER_BOMB = 17  # 癞子炸弹，由 ≥ 4 张癞子组成且必须有两种癞子同时存在
TYPE_PURE_JOKER_BOMB = 18  # 纯癞子炸弹，由 4 张同样点数的癞子组成
TYPE_HARD_BOMB = 19  # 硬炸弹，由 4 张同样点数的非癞子组成

# betting round action
PASS = 0
CALL = 1
RAISE = 2

# return all possible results of selecting num cards from cards list
def select(cards, num):
    return [list(i) for i in itertools.combinations(cards, num)]
