import collections

# 假设天癞子和地癞子可在此获取，实际应从环境正确传入
heaven_joker = None
earth_joker = None

def common_handle(moves, rival_move, key=lambda x: x[0]):
    """通用处理函数，支持自定义比较键"""
    rival_key = key(rival_move)
    new_moves = list()
    for move in moves:
        my_key = key(move)
        if my_key > rival_key:
            new_moves.append(move)
    return new_moves

def filter_type_1_single(moves, rival_move):
    return common_handle(moves, rival_move)

def filter_type_2_pair(moves, rival_move):
    return common_handle(moves, rival_move)

def filter_type_3_triple(moves, rival_move):
    return common_handle(moves, rival_move)

def filter_type_4_bomb(moves, rival_move):
    return common_handle(moves, rival_move)

# No need to filter for type_5_king_bomb

def filter_type_6_3_1(moves, rival_move):
    rival_move.sort()
    rival_rank = rival_move[1]
    new_moves = list()
    for move in moves:
        move.sort()
        my_rank = move[1]
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_7_3_2(moves, rival_move):
    rival_move.sort()
    rival_rank = rival_move[2]
    new_moves = list()
    for move in moves:
        move.sort()
        my_rank = move[2]
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_8_serial_single(moves, rival_move):
    return common_handle(moves, rival_move)

def filter_type_9_serial_pair(moves, rival_move):
    return common_handle(moves, rival_move)

def filter_type_10_serial_triple(moves, rival_move):
    return common_handle(moves, rival_move)

def filter_type_11_serial_3_1(moves, rival_move):
    rival = collections.Counter(rival_move)
    rival_rank = max([k for k, v in rival.items() if v == 3])
    new_moves = list()
    for move in moves:
        mymove = collections.Counter(move)
        my_rank = max([k for k, v in mymove.items() if v == 3])
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

# 新增软炸（TYPE_SOFT_BOMB）过滤函数
def filter_type_16_soft_bomb(moves, rival_move):
    def soft_bomb_key(move):
        non_joker = [c for c in move if c != heaven_joker and c != earth_joker]
        return (len(move), non_joker[0] if non_joker else 0)
    return common_handle(moves, rival_move, key=soft_bomb_key)

# 新增癞子炸弹（TYPE_JOKER_BOMB）过滤函数
def filter_type_17_joker_bomb(moves, rival_move):
    return common_handle(moves, rival_move, key=lambda x: len(x))

# 新增纯癞子炸弹（TYPE_PURE_JOKER_BOMB）过滤函数
def filter_type_18_pure_joker_bomb(moves, rival_move):
    return common_handle(moves, rival_move, key=lambda x: len(x))

# 新增硬炸弹（TYPE_HARD_BOMB）过滤函数
def filter_type_19_hard_bomb(moves, rival_move):
    return common_handle(moves, rival_move)
