import sys
import json
import time
import re
import string

import openai

from base import MMDAgentEXLabel


class ResponseGenerator:
    def __init__(self, config, asr_timestamp, query, dialogue_history, prompts):
        # 設定の読み込み
        self.max_tokens = config['ChatGPT']['max_tokens']
        self.max_message_num_in_context = config['ChatGPT']['max_message_num_in_context']
        self.model = config['ChatGPT']['response_generation_model']

        # 処理対象のユーザ発話に関する情報
        self.asr_timestamp = asr_timestamp
        self.query = query
        self.dialogue_history = dialogue_history
        self.prompts = prompts

        # 生成中の応答を保持・パースする変数
        self.response_fragment = ''
        self.punctuation_pattern = re.compile('[、。！？]')

        # ChatGPTに入力する対話文脈
        messages = []

        # 過去の対話履歴を対話文脈に追加
        i = max(0, len(self.dialogue_history) - self.max_message_num_in_context)
        messages.extend(self.dialogue_history[i:])

        # プロンプトおよび新しいユーザ発話を対話文脈に追加
        if query:
            messages.extend([
                {'role': 'user', 'content': self.prompts['RESP']},
                {'role': 'system', 'content': "OK"},
                {'role': 'user', 'content': query}
            ])
        # 新しいユーザ発話が存在せず自ら発話する場合のプロンプトを対話文脈に追加
        else:
            messages.extend([
                {'role': 'user', 'content': prompts['TO']}
            ])

        self.log(f"Call ChatGPT: {query=}")

        # ChatGPTに対話文脈を入力してストリーミング形式で応答の生成を開始
        self.response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=True
        )
    
    # Dialogueのsend_response関数で呼び出され，応答全体を一括で返し、/以降のラベルをパースする
    def __next__(self):
        def _parse_label(label_str):
            # 例: '0_平静,2_うなずく' → {"expression": ..., "action": ...}
            expression = MMDAgentEXLabel.id2expression[0]
            action = MMDAgentEXLabel.id2action[0]
            if "," in label_str:
                expr_part, act_part = label_str.split(",", 1)
                expr_id = expr_part.split("_")[0]
                act_id = act_part.split("_")[0]
                expr_id = int(expr_id) if expr_id.isdigit() else 0
                act_id = int(act_id) if act_id.isdigit() else 0
                expression = MMDAgentEXLabel.id2expression.get(expr_id, expression)
                action = MMDAgentEXLabel.id2action.get(act_id, action)
            return {"expression": expression, "action": action}

        # ChatGPTの応答をすべて結合
        for chunk in self.response:
            chunk_message = chunk['choices'][0]['delta']
            if 'content' in chunk_message.keys():
                self.response_fragment += chunk_message.get('content')
        # 応答全体を返す
        if self.response_fragment:
            result = self.response_fragment
            self.response_fragment = ''
            # '/' で発話とラベルを分割
            if '/' in result:
                phrase, label = result.rsplit('/', 1)
                label = label.strip()
                label_info = _parse_label(label)
                return {"phrase": phrase.strip(), **label_info}
            else:
                return {"phrase": result.strip()}
        raise StopIteration
    
    # ResponseGeneratorをイテレータ化
    def __iter__(self):
        return self

    # デバッグ用のログ出力
    def log(self, *args, **kwargs):
        print(f"[{time.time():.5f}]", *args, flush=True, **kwargs)


class ResponseChatGPT():
    def __init__(self, config, prompts):
        self.config = config
        self.prompts = prompts

        # 設定の読み込み
        openai.api_key = config['ChatGPT']['api_key']

        # 入力されたユーザ発話に関する情報を保持する変数
        self.user_utterance = ''
        self.response = ''
        self.last_asr_iu_id = ''
        self.asr_time = 0.0
    
    # ChatGPTの呼び出しを開始
    def run(self, asr_timestamp, user_utterance, dialogue_history, last_asr_iu_id, parent_llm_buffer):
        self.user_utterance = user_utterance
        self.last_asr_iu_id = last_asr_iu_id
        self.asr_time = asr_timestamp

        # ChataGPTを呼び出して応答の生成を開始
        self.response = ResponseGenerator(self.config, asr_timestamp, user_utterance, dialogue_history, self.prompts)

        # 自身をDialogueモジュールが持つLLMバッファに追加
        parent_llm_buffer.put(self)


if __name__ == "__main__":
    openai.api_key = '<enter your API key>'

    config = {'ChatGPT': {
        'max_tokens': 64,
        'max_message_num_in_context': 3,
        'response_generation_model': 'gpt-3.5-turbo'
    }}

    asr_timestamp = time.time()
    query = '今日は良い天気だね'
    dialogue_history = []
    prompts = {}

    with open('./prompt/response.txt') as f:
        prompts['RESP'] = f.read()

    response_generator = ResponseGenerator(config, asr_timestamp, query, dialogue_history, prompts)

    for part in response_generator:
        response_generator.log(part)
