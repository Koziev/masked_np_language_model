"""
Подготовка датасета для файнтюнинга ruT5 и ruGPT, чтобы модель могла подставлять NP в предложения.
Используется неразмеченный текст и синтаксический парсер UDPipe для выделения именных групп.

ATT: используются всякие локальные корпуса, которые я не выгружаю в общий доступ по разным соображениям.
Тем не менее, не вижу проблем с использованием любых других корпусов. См. функцию read_corpus1.
"""

import glob
import io
import os
import random

import pyconll
from ufal.udpipe import Model, Pipeline, ProcessingError

import extractors


def read_corpus1():
    """
    Чтение параграфов из одного большого корпуса.
    """
    with io.open('/home/inkoziev/corpora/Corpus/Raw/ru/text_blocks.txt', 'r', encoding='utf-8') as rdr:
        for line in rdr:
            # Возвращается абцаз из нескольких предложений, UDPipe будет сегментировать.
            yield line.strip()


def read_corpora2():
    """
    Чтение предложений из line-by-line корпусов в разных файлах
    """
    fnames = []
    dir1 = '/home/inkoziev/polygon/chatbot/data/SENTx'
    for filename in glob.iglob(dir1 + '/*.txt'):
        fnames.append(os.path.join(dir1, filename))

    dir2 = '/home/inkoziev/polygon/chatbot/data'
    for filename in ['facts5.txt', 'facts6.txt', 'facts7.txt', 'facts8.txt']:
        fnames.append(os.path.join(dir2, filename))

    sents = set()

    # Добавим предпосылок из QA датасета чатбота
    print('Loading pqa_all.dat')
    with io.open('/home/inkoziev/polygon/chatbot/tmp/pqa_all.dat', 'r', encoding='utf-8') as rdr:
        lines = []
        for line in rdr:
            s = line.strip()
            if s:
                lines.append(s)
            else:
                for premise in lines[:-2]:
                    sents.add(premise)
                lines.clear()

    for i, p in enumerate(fnames, start=1):
        print('Loading {}/{} file="{}"...'.format(i, len(fnames), p))
        with io.open(p, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                sents.add(line.strip())

    sents = sorted(sents, key=lambda z: random.random())

    return sents


def read_debug_corpus():
    return ['кошка хочет съесть мышку']


if __name__ == '__main__':
    # Скачать готовую модель для UDPipe можно тут https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131
    model = Model.load('/home/inkoziev/polygon/GramEval2020/tmp/udpipe_syntagrus.model')
    pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    udp_error = ProcessingError()

    print('Start parsing...')
    line_count = 0
    sample_count = 0
    with io.open('./data/t5_dataset.txt', 'w', encoding='utf-8') as wrt_t5, \
         io.open('./data/gpt_dataset.txt', 'w', encoding='utf-8') as wrt_gpt:
        #for line in read_corpus1():
        for line in read_corpora2():
        #for line in read_debug_corpus():
            line_count += 1
            if 0 == (line_count % 10000):
                # Время от времени показываем прогресс.
                print('{} lines, {} samples'.format(line_count, sample_count))
                if sample_count >= 100000:
                    # Ограничиваем размер тренировочного датасета
                    break

            # Выполняем синт. анализ очередного предложения
            processed = pipeline.process(line, udp_error)
            parsed_data = pyconll.load_from_string(processed)
            for parsing in parsed_data:
                if len(parsing) < 15:  # берем предложения длиной не более 15 токенов
                    for c_type, c_tokens in extractors.extract_constituents(parsing):
                        if 1 < len(c_tokens) < 5:  # слишком длинные составляющие пропускаем
                            # Собираем токены входного контекста, заменяя цепочку c_tokens на один <extra_id_0>
                            c_ids = [t.id for t in c_tokens]
                            input_tokens = []
                            for t in parsing:
                                if t.id == c_tokens[0].id:
                                    # Первый токен в NP
                                    input_tokens.append('<extra_id_0>')
                                elif t.id in c_ids:
                                    # Второй и последующие токены в составляющей пропускаем
                                    pass
                                else:
                                    input_tokens.append(t.form)

                            input_text = ' '.join(input_tokens)

                            # сэмпл для T5
                            output_text = '<extra_id_0>' + ' '.join(t.form for t in c_tokens)
                            wrt_t5.write('{}\t{}\n'.format(input_text, output_text))

                            # сэмпл для GPT
                            wrt_gpt.write('<s>{} # {}</s>\n'.format(input_text.replace('<extra_id_0>', '[{}]'.format(c_type)), ' '.join(t.form for t in c_tokens)))
                            sample_count += 1
