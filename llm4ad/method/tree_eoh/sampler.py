from __future__ import annotations

import re
from typing import Tuple, List, Dict

from .prompt import TreePrompt
from ...base import LLM, SampleTrimmer
from ...base.modify_code import ModifyCode
from .base.code import Function, Program


class TreeSampler:
    def __init__(self, llm: LLM, template_program: str | Program):
        self.llm = llm
        self._template_program = template_program

    def get_thought_and_function(self, prompt: str) -> Tuple[str, Function]:
        response = self.llm.draw_sample(prompt)
        thought = self.__class__.trim_thought_from_response(response)
        code = SampleTrimmer.trim_preface_of_function(response)

        function = SampleTrimmer.sample_to_function(code, self._template_program)
        return thought, function

    def get_multiple_thought_and_function(self, prompt: str) -> List[Tuple[str, Function]]:
        response = self.llm.draw_sample(prompt)
        idea_code_list = self.trim_multiple_idea_and_code_from_response(response)
        result = []
        for idea, code in idea_code_list:
            code = SampleTrimmer.trim_preface_of_function(code)
            function = SampleTrimmer.sample_to_function(code + '\n```', self._template_program)
            result.append((idea, function))
        return result

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        try:
            pattern = r'\{.*?\}'  # Compared with r'\{(.*)\}'
            bracketed_texts = re.findall(pattern, response)
            return bracketed_texts[0]
        except:
            return None

    @classmethod
    def trim_multiple_idea_and_code_from_response(cls, response: str) -> List[Tuple[str, str]]:
        matches = re.findall(r'<idea>(.*?)</idea>\s*<code>(.*?)</code>', response, re.DOTALL)

        if matches:
            # matches 已经是 [(idea_1, code_1), (idea_2, code_2), ...] 的列表形式
            # 这里进行简单的 strip 处理以去除首尾可能多余的换行符
            result = [(idea.strip(), code.strip()) for idea, code in matches]
            return result
        else:
            # 如果没有匹配到任何对，返回空列表（或者根据你的业务逻辑返回 None）
            return []
