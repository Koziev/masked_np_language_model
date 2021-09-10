"""
Вспомогательные функции для выделения NP, PP и прочих типов составляющих из разобранного предложения.
"""

import glob
import io
import os
import random

import pyconll
from ufal.udpipe import Model, Pipeline, ProcessingError


def compose_constituent(root_token, tokens, token2children):
    extracted_tokens = [root_token]
    extract_child_tokens(root_token, tokens, token2children, extracted_tokens)
    extracted_tokens = sorted(extracted_tokens, key=lambda t: int(t.id))

    # Убираем частицу в начале NP или PP:
    # проявляется и отношение к ним собирательницы
    #            ^^^
    if root_token.upos in ('NOUN', 'PROPN', 'PRON') and extracted_tokens[0].upos == 'PART' and extracted_tokens[0].form.lower() != 'не':
        extracted_tokens = extracted_tokens[1:]

    if extracted_tokens[-1].form in '.!?,:-':
        extracted_tokens = extracted_tokens[:-1]

    return extracted_tokens


def extract_child_tokens(root_token, tokens, token2children, extracted_tokens):
    for child in token2children.get(root_token.id, []):
        extracted_tokens.append(child)
        extract_child_tokens(child, tokens, token2children, extracted_tokens)


def extract_xcomp_without_obj(root_token, tokens, token2children, extracted_tokens):
    for child in token2children.get(root_token.id, []):
        if child.deprel != 'obj':
            extracted_tokens.append(child)
            extract_child_tokens(child, tokens, token2children, extracted_tokens)


def remove_final_punctuator(tokens):
    """
    Если цепочка токенов заканчивается пунктуатором, то убираем его.
    Это позволяет избавится от присоединения точек к финальной именной группе, к примеру.
    """
    if tokens[-1].form in '.,-!?;:…':
        return tokens[:-1]
    else:
        return tokens


def extract_nps(parsing):
    """
    Выделяет именные группы, возвращая за один раз одну ИГ.
    Возвращается цепочка токенов, входящих в именную группу.
    """
    token2children = dict()
    for token in parsing:
        if token.head != '0':
            if token.head in token2children:
                token2children[token.head].append(token)
            else:
                token2children[token.head] = [token]

    # Ищем глагольное сказуемое
    root = None
    for t in parsing:
        if t.deprel == 'root' and t.upos == 'VERB':
            root = t
            break

    if root:
        # Ищем существительные, присоединенные прямо к сказуемому
        for t in parsing:
            if t.upos in ('NOUN',) and t.head == root.id:
                # Исключаем PP, для этого проверим, что у существительного нет прикрепленного предлога.
                if not any((t2.upos == 'ADP' and t2.head == t.id) for t2 in parsing):
                    np_tokens = compose_constituent(t, parsing, token2children)
                    np_tokens = remove_final_punctuator(np_tokens)
                    yield np_tokens

        # Ищем существительные, входящие в PP
        for t in parsing:
            if t.upos in ('NOUN',):
                for t2 in token2children.get(t.id, []):
                    if t2.upos == 'ADP':
                        np_tokens = compose_constituent(t, parsing, token2children)
                        if np_tokens[0].id == t2.id: # исключаем случаи PP с префиксальной частицей "[и на старуху] бывает проруха"
                            # Уберем собственно предлог из цепочки токенов
                            np_tokens = [t3 for t3 in np_tokens if t3.id != t2.id]
                            np_tokens = remove_final_punctuator(np_tokens)
                            yield np_tokens


def extract_pps(parsing):
    """
    Выделяет предложные группы, возвращая за один раз одну PP.
    Возвращается цепочка токенов, входящих в PP.
    """
    token2children = dict()
    for token in parsing:
        if token.head != '0':
            if token.head in token2children:
                token2children[token.head].append(token)
            else:
                token2children[token.head] = [token]

    # Ищем глагольное сказуемое
    root = None
    for t in parsing:
        if t.deprel == 'root' and t.upos == 'VERB':
            root = t
            break

    if root:
        # Ищем существительные, входящие в PP
        for t in parsing:
            if t.upos in ('NOUN',):
                for t2 in token2children.get(t.id, []):
                    if t2.upos == 'ADP':
                        pp_tokens = compose_constituent(t, parsing, token2children)
                        pp_tokens = remove_final_punctuator(pp_tokens)
                        yield pp_tokens


def extract_vs(parsing):
    token2children = dict()
    for token in parsing:
        if token.head != '0':
            if token.head in token2children:
                token2children[token.head].append(token)
            else:
                token2children[token.head] = [token]

    # Ищем глагольное сказуемое
    root = None
    for t in parsing:
        if t.deprel == 'root' and t.upos == 'VERB':
            root = t
            break

    if root:
        # Проверим, что в предложении есть подлежащее или прямое дополнение
        sbj_found = any(t.deprel == 'nsubj' for t in parsing if t.head == root.id)
        obj_found = any(t.deprel == 'obj' for t in parsing if t.head == root.id)
        if sbj_found or obj_found:
            top_markup = []
            root_index = None
            for t in parsing:
                if t.deprel == 'root':
                    root_index = len(top_markup)
                    top_markup.append('root')
                elif t.head == root.id:
                    if t.deprel in ['advmod', 'obl', 'xcomp']:
                        top_markup.append(t.id)
                    else:
                        top_markup.append('bad_child')
                elif t.deprel == 'obj' and parsing[t.head].deprel == 'xcomp':
                    top_markup.append('bad_child')

            # Теперь ищем границы глагольной группы слева и справа от root
            start_index = None
            end_index = None
            for i in range(root_index, -1, -1):
                if top_markup[i] == 'bad_child':
                    break
                elif top_markup[i] == 'root':
                    start_index = root.id
                else:
                    start_index = top_markup[i]

            for i in range(root_index, len(top_markup)):
                if top_markup[i] == 'bad_child':
                    break
                elif top_markup[i] == 'root':
                    end_index = root.id
                else:
                    end_index = top_markup[i]

            start_index = int(start_index)
            end_index = int(end_index)

            # Теперь токены с start_index по end_index включительно представляют глагольную группу без разрывов.
            v_tokens = set()
            for t in parsing:
                if t.id == root.id:
                    v_tokens.add(t)
                else:
                    if start_index <= int(t.id) <= end_index:
                        tx = [t]
                        if t.deprel == 'xcomp':
                            extract_xcomp_without_obj(t, parsing, token2children, tx)
                        else:
                            extract_child_tokens(t, parsing, token2children, tx)

                        v_tokens.update(tx)

            v_tokens = remove_final_punctuator(sorted(v_tokens, key=lambda t: int(t.id)))
            yield v_tokens




def extract_constituents(parsing):
    for v in extract_vs(parsing):
        yield 'v1', v

    for np in extract_nps(parsing):
        yield 'np1', np

    for pp in extract_pps(parsing):
        yield 'pp1', pp
