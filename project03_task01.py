import os
from typing import List, Dict, Any
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
DEPARTMENTS = [
    "Департамент транспорта",
    "Департамент культуры",
    "Департамент здравоохранения",
    "Департамент образования",
    "Департамент экологии",
    "Департамент физической культуры и спорта"
]

# Конфигурация для локального API
LOCAL_API_BASE = "http://127.0.0.1:1337/v1"
LOCAL_API_MODEL = "qwen2.5:0.5b"

class CitizenRequest(BaseModel):
    """Модель данных обращения гражданина."""
    request_date: str
    request_topic: str
    target_department: str

class JsonSaveTool(BaseTool):
    """Инструмент для сохранения JSON в файл."""
    name: str = Field(default="json_save_tool")
    description: str = Field(default="Сохраняет данные обращения в JSON файл")

    def _load_existing_data(self) -> List[Dict]:
        """Загружает существующие данные из JSON файла."""
        try:
            with open("requests.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_data(self, data: List[Dict]):
        """Сохраняет данные в JSON файл."""
        with open("requests.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _run(self, tool_input: str = "", **kwargs) -> str:
        """Сохраняет новое обращение в JSON файл."""
        try:
            # Загружаем существующие данные
            existing_data = self._load_existing_data()
            
            # Преобразуем входные данные в словарь, если они переданы как строка
            if isinstance(tool_input, str):
                try:
                    request_data = json.loads(tool_input)
                except json.JSONDecodeError:
                    return "Ошибка: Неверный формат JSON"
            else:
                request_data = tool_input
            
            # Добавляем новые данные
            existing_data.append(request_data)
            
            # Сохраняем обновленные данные
            self._save_data(existing_data)
            
            return "Данные успешно сохранены"
            
        except Exception as e:
            return f"Ошибка при сохранении данных: {str(e)}"

class CitizenRequestClassifier:
    def __init__(self):
        """Инициализация классификатора обращений граждан."""
        self.llm = ChatOpenAI(
            base_url=LOCAL_API_BASE,
            model_name=LOCAL_API_MODEL,
            api_key="not-needed",  # Для локального API ключ не требуется
            temperature=0.01
        )
        
        # Инициализация инструмента сохранения JSON
        self.json_tool = JsonSaveTool()
        
        # Создание шаблона промпта
        template = """Ты - система классификации обращений граждан. Твоя задача - определить, в какой департамент направить обращение.

ПРАВИЛА КЛАССИФИКАЦИИ:
1. Анализируй ОСНОВНУЮ проблему, игнорируя второстепенные детали
2. Обращай внимание на ключевые слова и контекст
3. Выбирай ТОЛЬКО ОДИН департамент из списка
4. Если обращение затрагивает несколько сфер, выбери департамент по главной проблеме
5. Используй точное название департамента из списка

ВАЖНО! Определение департамента по темам:
- Шум, загрязнение, экология -> Департамент экологии
- Общественный транспорт, маршруты, тарифы -> Департамент транспорта
- Дороги, тротуары, светофоры -> Департамент транспортной инфраструктуры
- Культура, досуг, искусство -> Департамент культуры
- Здоровье, лечение, медицина -> Департамент здравоохранения
- Обучение, школы, образование -> Департамент образования
- Спорт, физкультура, соревнования -> Департамент физической культуры и спорта

ПРИМЕРЫ КЛАССИФИКАЦИИ ПО ДЕПАРТАМЕНТАМ:

1. Департамент транспорта:
Примеры обращений:
- "Регулярные отмены поездок трамваев маршрута №17 и нарушения графика движения поездов метро. Люди вынуждены ожидать значительное время, пропускают важные дела."
- "Постоянно сталкиваюсь с проблемой переполненности пригородных электропоездов утром и вечером. В вагоны невозможно войти."
- "Нет четких схем маршрутов наземного транспорта на остановках. Возникает путаница с выбором нужного рейса."
- "Не устраивает система оплаты проезда, высокие тарифы при ограниченных возможностях пользования проездными билетами."
- "Электротранспорт регулярно ломается, кабины грязные, сиденья неисправны. Стало невыносимо пользоваться таким транспортом."
- "Живу в отдаленном микрорайоне, где практически отсутствует транспортное сообщение с центром города."

2. Департамент культуры:
Примеры обращений:
- "Выражаем недоумение по поводу закрытия нашего любимого Театра юного зрителя. Просим объяснить причины."
- "Испытываем дефицит культурной жизни. Просим активизировать выездные спектакли театров и кинопоказы."
- "Районная библиотека в плачевном состоянии: старые книги, устаревшие компьютеры, недостаток литературы."
- "В музее истории города обнаружены трещины на стенах и крыше, окна требуют замены."
- "Фестивали проводятся формально, интерес публики минимален. Просим привлекать молодые таланты."
- "Исторически значимые памятники нашей области разрушаются и нуждаются в срочном ремонте."

3. Департамент здравоохранения:
Примеры обращений:
- "Огромные очереди в поликлинике №23. Ждать приема врача приходится несколько часов."
- "Дефицит врачей-стоматологов и офтальмологов в поликлинике №15. Жители месяцами ждут очереди."
- "Аппаратура в поликлинике старая, диагностика затруднительна. Аппарат УЗИ не работает полгода."
- "Сельчане сталкиваются со сложностями при обращении за неотложной медпомощью. Скорая помощь приезжает с задержкой."
- "Приобретение жизненно необходимых лекарств стало проблемой. Аптеки сообщают о дефиците."
- "Детская больница в ужасном состоянии: стены облезлые, санузел в антисанитарии, кровати скрипят."

4. Департамент образования:
Примеры обращений:
- "Прошу перевести ребенка из школы №15 в школу №23 в связи с переездом семьи."
- "Плохое качество школьных обедов в гимназии №32. Питание однообразное, порции маленькие."
- "Учителя подвергаются унижениям со стороны учеников старших классов."
- "Компьютерный класс оснащен старыми ПК. Учащиеся не могут изучать современные технологии."
- "Учебники имеют много ошибок, страницы вырваны, обложки повреждены."
- "Старое здание школы серьезно повреждено: течет крыша, стены с плесенью, лестницы изношены."

5. Департамент экологии:
Примеры обращений:
- "Загрязнение реки пластиковыми бутылками и мусором. Рыбалка и отдых стали невозможны."
- "Началась вырубка зеленых насаждений без согласования с жильцами."
- "Увеличение выброса дыма и неприятных запахов с завода металлургии."
- "В лесопарковой зоне обнаружили несанкционированную свалку бытовых отходов."
- "Самолеты часто пролетают над домом, издавая сильный шум."
- "Утилизация батареек и ртутьсодержащих ламп осуществляется неправильно."

6. Департамент физической культуры и спорта:
Примеры обращений:
- "Просьба оборудовать спортивную площадку. Детям негде активно отдыхать, тренажеры сломаны."
- "Спортивный зал школы в аварийном состоянии: потолок протекает, пол прогибается."
- "Жители готовы участвовать в соревнованиях по футболу и волейболу, но мероприятия не проводятся."
- "Просим открыть секцию художественной гимнастики для девочек."
- "Нехватка длинных качественных лыжных трасс."
- "Команда по мини-футболу нуждается в новой форме и футбольных бутсах."

7. Департамент транспортной инфраструктуры:
Примеры обращений:
- "Опасный переход через железную дорогу, сложно переходить путь детям и пенсионерам."
- "Покрытие улиц на маршрутах сильно ухудшилось, происходят поломки транспорта."
- "Отсутствие пандусов и неровности тротуаров создают сложности для инвалидов."
- "Требуется ремонт асфальта на улице."
- "Нужен светофор на опасном перекрестке."
- "Остановка требует оборудования пандусом для инвалидов."

Проанализируй следующее обращение и верни ТОЛЬКО название соответствующего департамента:
{request}"""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["request"]
        )
        
        # Создание цепочки обработки
        self.chain = (
            {"request": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )

    def classify_request(self, request_text: str) -> CitizenRequest:
        """
        Классифицирует обращение гражданина и определяет целевой департамент.
        
        Args:
            request_text (str): Текст обращения
            
        Returns:
            CitizenRequest: Структурированный ответ с информацией об обращении
        """
        try:
            response = self.chain.invoke(request_text)
            
            # Получаем текст ответа
            department = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # Создаем объект CitizenRequest
            result = CitizenRequest(
                request_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                request_topic=request_text[:100] + "..." if len(request_text) > 100 else request_text,
                target_department=department
            )
            
            # Сохраняем результат в файл
            self.json_tool._run(result.model_dump())
            
            return result
            
        except Exception as e:
            print(f"\nОшибка при классификации: {e}")
            result = CitizenRequest(
                request_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                request_topic="Не удалось определить тему",
                target_department="Не определено"
            )
            
            # Сохраняем даже ошибочный результат
            self.json_tool._run(result.model_dump())
            
            return result

def main():
    """Основная функция для работы с пользовательским вводом."""
    print("\nДобро пожаловать в систему классификации обращений граждан!")
    print("=" * 60)
    print("\nДля выхода введите 'exit' или нажмите Ctrl+C")
    print("-" * 60)
    
    # Инициализация классификатора
    try:
        classifier = CitizenRequestClassifier()
    except Exception as e:
        print(f"Ошибка при инициализации системы: {e}")
        return
    
    while True:
        try:
            # Получение обращения от пользователя
            print("\nВведите ваше обращение:")
            request_text = input("> ").strip()
            
            # Проверка на выход
            if request_text.lower() == 'exit':
                print("\nЗавершение работы...")
                break
            
            # Проверка на пустой ввод
            if not request_text:
                print("Обращение не может быть пустым. Попробуйте еще раз.")
                continue
            
            # Классификация обращения
            print("\nОбрабатываем ваше обращение...")
            result = classifier.classify_request(request_text)
            
            # Вывод результата
            print("\nРезультат классификации:")
            print("-" * 40)
            print(f"Дата: {result.request_date}")
            print(f"Текст обращения: {result.request_topic}")
            print(f"Определенный департамент: {result.target_department}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\nЗавершение работы...")
            break
        except Exception as e:
            print(f"\nПроизошла ошибка: {e}")
            print("Пожалуйста, попробуйте еще раз.")

if __name__ == "__main__":
    main() 