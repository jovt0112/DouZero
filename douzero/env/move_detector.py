from douzero.env.utils import *
import collections

# check if move is a continuous sequence
def is_continuous_seq(move):
    i = 0
    while i < len(move) - 1:
        if move[i+1] - move[i] != 1:
            return False
        i += 1
    return True

# return the type of the move
def get_move_type(move, heaven_joker, earth_joker):  # 添加天癞子和地癞子作为参数
    move_size = len(move)
    move_dict = collections.Counter(move)

    if move_size == 0:
        return {'type': TYPE_0_PASS}

    if move_size == 1:
        return {'type': TYPE_1_SINGLE, 'rank': move[0]}

    if move_size == 2:
        if move[0] == move[1]:
            return {'type': TYPE_2_PAIR, 'rank': move[0]}
        elif move == [20, 30]:  # Kings
            return {'type': TYPE_5_KING_BOMB}
        else:
            return {'type': TYPE_15_WRONG}

    if move_size == 3:
        if len(move_dict) == 1:
            return {'type': TYPE_3_TRIPLE, 'rank': move[0]}
        else:
            return {'type': TYPE_15_WRONG}

    if move_size == 4:
        # 检查硬炸弹（非癞子）
        if len(move_dict) == 1 and move[0] != heaven_joker and move[0] != earth_joker:
            return {'type': TYPE_HARD_BOMB, 'rank': move[0]}
        # 检查纯癞子炸弹
        elif len(move_dict) == 1 and (move[0] == heaven_joker or move[0] == earth_joker):
            return {'type': TYPE_PURE_JOKER_BOMB, 'rank': move[0]}
        # 检查软炸（这里简单判断有癞子且非上述情况，实际需更精确逻辑）
        elif any(c == heaven_joker or c == earth_joker for c in move) and len(move_dict) <= 2:
            return {'type': TYPE_SOFT_BOMB, 'rank': move[0]}
        elif len(move_dict) == 2:
            if move[0] == move[1] == move[2] or move[1] == move[2] == move[3]:
                return {'type': TYPE_6_3_1, 'rank': move[1]}
            else:
                return {'type': TYPE_15_WRONG}
        else:
            return {'type': TYPE_15_WRONG}

    # 检查癞子炸弹（≥4张癞子且有两种）
    if move_size >=4 and all(c == heaven_joker or c == earth_joker for c in move) and len(set([c for c in move if c == heaven_joker or c == earth_joker])) == 2:
        return {'type': TYPE_JOKER_BOMB, 'len': move_size}

    if is_continuous_seq(move):
        return {'type': TYPE_8_SERIAL_SINGLE, 'rank': move[0], 'len': len(move)}

    if move_size == 5:
        if len(move_dict) == 2:
            return {'type': TYPE_7_3_2, 'rank': move[2]}
        else:
            return {'type': TYPE_15_WRONG}

    count_dict = collections.defaultdict(int)
    for c, n in move_dict.items():
        count_dict[n] += 1

    if move_size == 6:
        if (len(move_dict) == 2 or len(move_dict) == 3) and count_dict.get(4) == 1 and \
                (count_dict.get(2) == 1 or count_dict.get(1) == 2):
            return {'type': TYPE_13_4_2, 'rank': move[2]}

    if move_size == 8 and (((len(move_dict) == 3 or len(move_dict) == 2) and
            (count_dict.get(4) == 1 and count_dict.get(2) == 2)) or count_dict.get(4) == 2):
        return {'type': TYPE_14_4_22, 'rank': max([c for c, n in move_dict.items() if n == 4])}

    mdkeys = sorted(move_dict.keys())
    if len(move_dict) == count_dict.get(2) and is_continuous_seq(mdkeys):
        return {'type': TYPE_9_SERIAL_PAIR, 'rank': mdkeys[0], 'len': len(mdkeys)}

    if len(move_dict) == count_dict.get(3) and is_continuous_seq(mdkeys):
        return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': mdkeys[0], 'len': len(mdkeys)}

    # Check Type 11 (serial 3+1) and Type 12 (serial 3+2)
    if count_dict.get(3, 0) >= MIN_TRIPLES:
        serial_3 = list()
        single = list()
        pair = list()

        for k, v in move_dict.items():
            if v == 3:
                serial_3.append(k)
            elif v == 1:
                single.append(k)
            elif v == 2:
                pair.append(k)
            else:  # no other possibilities
                return {'type': TYPE_15_WRONG}

        serial_3.sort()
        if is_continuous_seq(serial_3):
            if len(serial_3) == len(single)+len(pair)*2:
                return {'type': TYPE_11_SERIAL_3_1, 'rank': serial_3[0], 'len': len(serial_3)}
            if len(serial_3) == len(pair) and len(move_dict) == len(serial_3) * 2:
                return {'type': TYPE_12_SERIAL_3_2, 'rank': serial_3[0], 'len': len(serial_3)}

        if len(serial_3) == 4:
            if is_continuous_seq(serial_3[1:]):
                return {'type': TYPE_11_SERIAL_3_1, 'rank': serial_3[1], 'len': len(serial_3) - 1}
            if is_continuous_seq(serial_3[:-1]):
                return {'type': TYPE_11_SERIAL_3_1, 'rank': serial_3[0], 'len': len(serial_3) - 1}

    return {'type': TYPE_15_WRONG}
