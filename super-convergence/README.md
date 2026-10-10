# Super-Convergence

Воспроизведение CLR из таблицы 6 [статьи Smith и Topin](https://arxiv.org/html/1708.07120v3)
на PyTorch. Сравниваем его с нашим step LR на ResNet-56 и CIFAR-10.

[Отчёт](report.pdf) · [Презентация](presentation.pptx) ·
[Презентация в PDF](presentation.pdf) · [Статья](paper.pdf).

## Результат

| Режим | Шаги | Test accuracy | Время обучения |
|---|---:|---:|---:|
| CLR | 2000 | 90,90% | 11 мин 48 с |
| Step LR | 4000 | 83,93% | 23 мин 54 с |

В нашем сравнении разница accuracy составила **6,97 процентного пункта**.
В таблице 6 статьи для CLR указано 89,7% за 2000 шагов, у нас — 90,90% за те же 2000 шагов.

Accuracy измеряли каждые 100 шагов на всех 10 000 test-примерах.
Время включает подачу данных и синхронизированные шаги обучения.

## Условия опыта

| Параметр | CLR (`paper`) | Step LR (`step`) |
|---|---|---|
| Шаги и эпохи | 2000 шагов, 40 эпох | 4000 шагов, 80 эпох |
| LR schedule | 0,1 → 3 → 0,1, пик на индексе 1000 | 0,1; 0,01 после 2000; 0,001 после 3000 |
| Модель и данные | ResNet-56, 50 000 train / 10 000 test | То же |
| Batch / microbatch | 1000 / 125 | То же |
| BN fraction / momentum / weight decay | 0,80 / 0,9 / 0,0001 | То же |
| Seed | 0 | 0 |
| GPU | NVIDIA L4 | NVIDIA L4 |

Начальные веса, порядок данных и случайные горизонтальные отражения одинаковые.
Из изображений вычитали среднее по train. Для оценки использовали статистики BN,
накопленные во время обучения. В каждом режиме выполнен один запуск с seed 0.

CLR взят из таблицы 6 и раздела `TAB1b`
[кода авторов](https://github.com/lnsmith54/super-convergence/blob/ecc561594b39442661e7e0a39d2820b0f4147324/x.sh).
Step LR — наш сокращённый schedule на 4000 шагов. Параметры BN и weight decay
в обоих режимах одинаковые. Порог 90% и длину step LR выбрали после просмотра CLR,
до запуска step LR.

## Как повторить

Использованы Python 3.12.15, PyTorch 2.11.0+cu130 и NVIDIA L4.
Команды из папки `super-convergence`, в окружении с поддержкой CUDA:

```bash
python -m pip install -r requirements.txt
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python train.py --method paper --seed 0 --device cuda --workers 4 --threads 4 --deterministic
python train.py --method step --seed 0 --device cuda --workers 4 --threads 4 --deterministic
python plot.py runs/paper runs/step --output results/repeat
```

`train.py` содержит модель, данные, оптимизатор и обучение. `plot.py` строит графики.
Настройки находятся в `SETTINGS` и `METHODS`, по умолчанию выбран `paper`.
В PyTorch перенесены архитектура авторов, правило CaffeSGD и накопление статистик BN.

Данные загружаются в `data/`. В `runs/<method>/` сохраняются веса, среднее изображение,
метаданные и `metrics.jsonl`. Обучение начинается с нуля. Для отдельного повтора
укажите новый `--output`, иначе файлы перезапишутся. `--stop-after 100` выполняет
короткую проверку с исходным schedule. На Mac можно указать `--device mps`.

## Результаты запусков

- [CLR: метрики, настройки и окружение](results/gcp-l4-table6-2000-20261010/).
- [Step LR: метрики, настройки и окружение](results/gcp-l4-speed-4000-20261010/).
- [Сравнение запусков](results/gcp-l4-speed-4000-20261010/comparison.json).
- [Графики](results/gcp-l4-speed-4000-20261010/curves.png).
