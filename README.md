Проект: Volleyball Action Annotator (VAA)
Цель: Инструмент разметки волейбольного матча

Размечаем начало и конец игрового действия
Поддерживаем 3-frame grayscale → RGB (суперкадр)
Пользователь видит обычный кадр, но может переключиться на суперкадр
Сохраняем разметку в YOLO-формате (по кадрам) + временные метки действий


Serve, Reception, Set, Attack, Block, Dig

## State Machine

```mermaid
stateDiagram-v2
    [*] --> BEFORE_SERVE
    BEFORE_SERVE --> AFTER_SERVE: подача выполнена успешно
    BEFORE_SERVE --> RALLY_END: ошибка подачи

    AFTER_SERVE --> AFTER_RECEIVE: успешный/плохой/переход приём
    AFTER_SERVE --> RALLY_END: ошибка приёма

    AFTER_RECEIVE --> AFTER_SET: передача (успешная/ошибка)
    AFTER_RECEIVE --> FREEBALL: переходящий/без атаки приём

    AFTER_SET --> IN_ATTACK: попытка атаки / атака
    AFTER_SET --> RALLY_END: ошибка передачи

    IN_ATTACK --> RALLY_END: очко атака / ошибка атака / блок / блок-аут
    IN_ATTACK --> AFTER_RECEIVE: защита удалась (продолжение розыгрыша)

    RALLY_END --> BEFORE_SERVE: новый розыгрыш
```

```
src/
├── core/
│   ├── video_processor.py        ← Логика 3-frame RGB
│   ├── annotation_manager.py     ← Управление действиями и YOLO-боксами
│   └── yolo_tracker.py           ← Обёртка над YOLO
│
├── ui/
│   ├── image_canvas.py           ← (существующий) + улучшения
│   ├── main_window.py            ← Основное окно
│   └── widgets/
│       ├── timeline.py
│       └── action_panel.py
│
├── config/
│   └── config.py                 ← AnnotationConfig + UI настройки
│
└── utils/
    └── exporters.py              ← JSON, YOLO txt, CSV
```

