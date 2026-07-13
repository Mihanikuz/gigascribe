# autoreplacer.py

import json
import re
from typing import List, Dict, Any, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoReplacer:
    """
    Класс для автоматической замены текста на основе правил из JSON-файла.
    
    Поддерживаемые режимы:
    - literal: Простая замена подстроки
    - regex: Замена с использованием регулярных выражений
    
    Поддерживаемые цели:
    - transcript: Применять к транскрипции
    - summary: Применять к резюме/сводке
    """

    def __init__(self, rules_file: str = "autoreplace.json"):
        """
        Инициализация AutoReplacer.
        
        Args:
            rules_file (str): Путь к файлу с правилами автозамены
        """
        self.rules_file = rules_file
        self.rules = []
        self.load_rules()
    
    def load_rules(self):
        """
        Загрузка правил автозамены из JSON-файла.
        """
        try:
            with open(self.rules_file, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            
            # Валидация и сортировка правил по приоритету
            validated_rules = []
            for rule in rules_data:
                if self._validate_rule(rule):
                    validated_rules.append(rule)
                else:
                    logger.warning(f"Некорректное правило пропущено: {rule}")
            
            # Сортировка по убыванию приоритета
            self.rules = sorted(validated_rules, key=lambda x: x.get('priority', 0), reverse=True)
            logger.info(f"Загружено {len(self.rules)} правил автозамены")
            
        except FileNotFoundError:
            logger.error(f"Файл с правилами {self.rules_file} не найден")
            self.rules = []
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON в файле {self.rules_file}: {e}")
            self.rules = []
        except Exception as e:
            logger.error(f"Неожиданная ошибка при загрузке правил: {e}")
            self.rules = []
    
    def _validate_rule(self, rule: Dict[str, Any]) -> bool:
        """
        Валидация правила автозамены.
        
        Args:
            rule (Dict[str, Any]): Правило для валидации
            
        Returns:
            bool: True, если правило валидно, иначе False
        """
        required_fields = ['pattern', 'replace', 'mode', 'targets']
        for field in required_fields:
            if field not in rule:
                logger.warning(f"В правиле отсутствует обязательное поле: {field}")
                return False
        
        if rule['mode'] not in ['literal', 'regex']:
            logger.warning(f"Неподдерживаемый режим: {rule['mode']}")
            return False
            
        if not isinstance(rule['targets'], list):
            logger.warning("Поле 'targets' должно быть списком")
            return False
            
        return True
    
    def _prepare_pattern(self, pattern: str, mode: str, case_sensitive: bool, whole_word: bool) -> re.Pattern:
        """
        Подготовка паттерна для поиска.
        
        Args:
            pattern (str): Исходный паттерн
            mode (str): Режим поиска (literal/regex)
            case_sensitive (bool): Учитывать регистр
            whole_word (bool): Только целые слова
            
        Returns:
            re.Pattern: Скомпилированный паттерн
        """
        if mode == 'literal':
            # Экранируем специальные символы для literal режима
            pattern = re.escape(pattern)
        
        if whole_word:
            pattern = r'\b' + pattern + r'\b'
        
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.compile(pattern, flags)
    
    def apply_rules(self, text: str, target: str = "transcript") -> str:
        """
        Применение правил автозамены к тексту.
        
        Args:
            text (str): Исходный текст
            target (str): Цель применения (transcript/summary)
            
        Returns:
            str: Текст после применения правил
        """
        if not text:
            return text
            
        result_text = text
        
        for rule in self.rules:
            # Проверяем, применимо ли правило к данной цели
            if target not in rule.get('targets', []):
                continue
            
            try:
                pattern = self._prepare_pattern(
                    rule['pattern'],
                    rule['mode'],
                    rule.get('case_sensitive', False),
                    rule.get('whole_word', False)
                )
                
                result_text = pattern.sub(rule['replace'], result_text)
                
            except re.error as e:
                logger.error(f"Ошибка в регулярном выражении правила: {rule['pattern']}. Ошибка: {e}")
            except Exception as e:
                logger.error(f"Ошибка применения правила {rule['pattern']}: {e}")
        
        return result_text
    
    def apply_to_transcript(self, transcript_text: str) -> str:
        """
        Применение правил к транскрипции.
        
        Args:
            transcript_text (str): Текст транскрипции
            
        Returns:
            str: Транскрипция после применения правил
        """
        return self.apply_rules(transcript_text, "transcript")
    
    def apply_to_summary(self, summary_text: str) -> str:
        """
        Применение правил к резюме/сводке.
        
        Args:
            summary_text (str): Текст резюме/сводки
            
        Returns:
            str: Резюме/сводка после применения правил
        """
        return self.apply_rules(summary_text, "summary")
    
    def reload_rules(self):
        """
        Перезагрузка правил из файла.
        """
        logger.info("Перезагрузка правил автозамены")
        self.load_rules()

# Глобальный экземпляр для удобства использования
autoreplacer = AutoReplacer()

# Функции для удобного использования
def apply_autoreplace_transcript(text: str) -> str:
    """Применить автозамену к транскрипции"""
    return autoreplacer.apply_to_transcript(text)

def apply_autoreplace_summary(text: str) -> str:
    """Применить автозамену к резюме/сводке"""
    return autoreplacer.apply_to_summary(text)

def reload_autoreplace_rules():
    """Перезагрузить правила автозамены"""
    autoreplacer.reload_rules()

# Пример использования:
if __name__ == "__main__":
    # Создание экземпляра
    replacer = AutoReplacer("autoreplace.json")
    
    # Тестовый текст
    test_text = "транспривод тверь и епкшные системы работают с ржд"
    
    # Применение к транскрипции
    result_transcript = replacer.apply_to_transcript(test_text)
    print(f"Транскрипция: {result_transcript}")
    
    # Применение к резюме
    result_summary = replacer.apply_to_summary(test_text)
    print(f"Резюме: {result_summary}")