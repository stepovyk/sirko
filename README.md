# Сірко

*A chess engine written from scratch in Rust: bitboards, PVS search with Lazy SMP, its own NNUE trained on its own self-play with its own trainer. No code from other engines, no dependencies. Plays on Lichess as [SirkoAI](https://lichess.org/@/SirkoAI). English summary at the end.*

Шаховий рушій, написаний з нуля на Rust, без жодного чужого рушійного коду. Грає на Lichess як [SirkoAI](https://lichess.org/@/SirkoAI). Назва – на честь кошового Івана Сірка, найвідомішого характерника.

## Що всередині

- **Дошка** – бітборди з власними магічними числами, які рушій шукає сам на старті. Хід – 16 біт, позиція копіюється на кожен хід. Класичні шахи й шахи Фішера працюють однаково: рокіровка записується як «король бере свою туру».
- **Пошук** – ітеративне поглиблення з вікнами прагнення, PVS і спільна таблиця переставлень на кількох потоках (Lazy SMP) без блокувань. Також нульовий хід, скорочення пізніх ходів з урахуванням історії, відсічення марних ходів, однинні (й подвійні) продовження, тихий пошук із SEE, ходи-вбивці, історія ходів і продовжень, контрходи й поправка оцінки за історією пошуку.
- **Оцінка** – власна нейромережа (нижче). Запасна – ручна оцінка з вагами для середини гри й закінчення: матеріал, положення фігур, рухливість, пішаки, безпека короля, загрози.
- **UCI** – обдумування на час суперника (ponder і ponderhit), шахи Фішера, Hash, Threads, Move Overhead, Clear Hash і EvalFile. Цього досить для lichess-bot.

## Мережа

Самогра (`sirko datagen ПАРТІЙ ВУЗЛІВ ПОТОКІВ ФАЙЛ [ХВИЛИН]`) пише спокійні позиції з оцінкою пошуку й результатом партії. Власний тренер (`sirko train ВИХІД ПРИХОВАНИХ ЕПОХ [--batch --lr --wdl --buckets] ДАНІ...`) вчить мережу (768 → H)×2 → 1 зі SCReLU і окремим виходом залежно від кількості фігур, а рушій оцінює нею позиції з покроковим оновленням накопичувачів. Мережа з `nets/sirko.nnue` вбудовується при збиранні, а `setoption name EvalFile` дає перевірити іншу без перезбирання.

Тренер читає дані з диска частинами, тож обсяг даних не впирається в пам'ять. З `--state=ФАЙЛ --max-minutes=N` він після кожної епохи зберігає весь стан і не починає епоху, що не встигне: навчання в GitHub Actions (`train.yml`) іде до трьох етапів по ~5,5 години, і кожен продовжує попередній так, ніби перерви не було.

Сірко вчиться сам щотижня: самогра поточною мережею, навчання на ній і на свіжій самогрі попередніх тижнів, матч проти поточної мережі на 4800 партій. Якщо нова мережа впевнено сильніша (імовірність ≥ 97,5%), вона сама йде в рушій, а сервер ставить її наступної ночі. Окремо щотижня – замір сили проти свіжого Stockfish.

## Перевірки

```bash
cargo test --release                                       # perft, SEE, таблиця, симетрія оцінки, накопичувачі мережі, тренер
./target/release/sirko perft 6                             # 119060324
python3 tools/movegen_check.py ./target/release/sirko      # генерація ходів проти python-chess
python3 tools/uci_check.py ./target/release/sirko          # протокол, мати, ponder, шахи Фішера, партія сам із собою
python3 tools/match.py --a ./target/release/sirko --b stockfish --b-opt UCI_LimitStrength=true --b-opt UCI_Elo=2800 --games 30 --tc 8+0.08
```

## Сила

Проти Stockfish з обмеженою силою, контроль 8+0,08 с, один потік:

| Суперник | Результат |
|---|---|
| Stockfish 17.1, Elo 2200 (версія 0.1, ручна оцінка) | 20 з 20 |
| Stockfish 17.1, Elo 2800 (версія 0.1, ручна оцінка) | 14,5 з 30 (+10 =9 −11) |
| Stockfish 19, Elo 3190 (найвища, яку він дозволяє) | 52,4% очок, +801 =912 −687 (2400 партій) |

Кожна зміна йде в рушій, лише якщо виграла матч у попередньої версії (GitHub Actions, сотні партій):

| Зміна | Результат |
|---|---|
| Історія продовжень і взять | +14 ± 14 Elo (2400 партій, 5+0,05 с) |
| Власна мережа (256 нейронів, 54 млн позицій самогри) | +195 ± 28 Elo проти ручної оцінки (800 партій, 8+0,08 с) |
| Мережа другого покоління (86 млн позицій самогри вже з мережею, 8 виходів) | +67 ± 25 Elo проти першої мережі (800 партій, 8+0,08 с) |
| Числа пошуку з підбору SPSA (16 ланцюжків по 5 годин) | +94 ± 20 Elo (1200 партій, 8+0,08 с) |
| Історія ходів живе від ходу до ходу всю партію | +32 ± 24 Elo (800 партій, 8+0,08 с) |
| Мережа третього покоління (177 млн позицій самогри) | +49 ± 24 Elo проти другої (800 партій, 8+0,08 с) |
| Мережа четвертого покоління (265 млн позицій самогри) | +19 ± 20 Elo проти третьої (1200 партій, 8+0,08 с) |
| Другий підбір SPSA (10 ланцюжків по 5 годин) | +24 ± 14 Elo (2400 партій, 8+0,08 с) |
| Третій підбір SPSA (12 ланцюжків по 5 годин) | +30 ± 20 Elo (1200 партій, 8+0,08 с) |
| Швидша мережа: оновлення за один прохід, вихід через pmaddwd, відкладені накопичувачі | +32 ± 20 Elo (1200 партій, 8+0,08 с) |
| Мережа п'ятого покоління (317 млн позицій самогри) | +11 ± 14 Elo проти четвертої (2400 партій, 8+0,08 с) |
| Корекція статичної оцінки за пішаками й фігурами кожного кольору | +9 ± 10 Elo (4800 партій, 8+0,08 с) |
| Мережа шостого покоління (395 млн позицій самогри) | +11 ± 14 Elo проти п'ятої (2400 партій, 8+0,08 с) |
| Глибший чи мілкіший повторний пошук після скорочення, більше скорочення тихих ходів, коли хід з таблиці – взяття | +10 ± 10 Elo (4800 партій, 8+0,08 с) |
| Ширша мережа: 384 нейрони замість 256 (ті самі 395 млн позицій) | +18 ± 14 Elo (2400 партій, 8+0,08 с) |
| Подвійне розширення явно єдиного ходу, мілкіше – коли хід не єдиний | +10 ± 10 Elo (4800 партій, 8+0,08 с) |
| Мережа сьомого покоління (707 млн позицій самогри, навчання частинами з диска) | +16 ± 10 Elo проти попередньої (4800 партій, 8+0,08 с) |
| Ширша мережа: 512 нейронів, 972 млн позицій, 10 епох етапами | +22 ± 14 Elo проти попередньої (2400 партій, 8+0,08 с) |

Мережа вбудована в рушій, ручна оцінка лишилася запасною.

## Бот SirkoAI

Сірко грає на Lichess окремим ботом [SirkoAI](https://lichess.org/@/SirkoAI) на власному сервері (Ubuntu чи Debian, зокрема ARM) через lichess-bot. Думає лише сам рушій: ні баз дебютів, ні хмарного аналізу; коли на дошці до 7 фігур, ходить ідеально за таблицями закінчень Lichess. Рушій бере всі ядра сервера, грає одну партію за раз і нікого не уникає: приймає виклики від усіх і сам кличе ботів, коли без гри.

Встановлення:

```bash
sudo git clone https://github.com/stepovyk/sirko.git /opt/sirko/src && sudo /opt/sirko/src/deploy/sirko-bot install
sudo sirko-bot token
```

Перша команда ставить усе (свіжий lichess-bot, рушій, зібраний і перевірений на місці, служби), друга питає ключ Lichess акаунта SirkoAI з дозволом «Play games with the bot API» і запускає бота. Ключ лишається лише на сервері.

Далі сервер щоночі сам ставить нову версію з гілки `release`, куди CI переносить лише перевірені коміти. Керування: `sirko-bot help`.

## English

Sirko is a UCI chess engine written from scratch in Rust with no dependencies and no code taken from other engines.

- **Board**: bitboards with magic numbers found at startup, copy-make, standard chess and Chess960 (castling encoded as king-takes-rook).
- **Search**: iterative deepening with aspiration windows, PVS, lock-free shared transposition table (Lazy SMP), null move, history-aware LMR, futility and late-move pruning, singular and double extensions, SEE-pruned quiescence, killers, history and continuation history, countermoves, correction history.
- **Evaluation**: its own NNUE ((768 → 512)×2 → 1 with output buckets by piece count, SCReLU, int16, incremental accumulators), trained by its own trainer on its own self-play; a hand-written tapered evaluation is kept as a fallback.
- **Self-improvement**: every week GitHub Actions runs self-play with the current net, trains a new one on about a billion positions in resumable stages, and merges it only if it wins a 4800-game match with ≥ 97.5% likelihood of superiority.
- **Verification**: perft and a differential move generator test against python-chess on tens of thousands of positions, UCI protocol checks, CI on x86 and ARM.

Build: `cargo build --release`. License: GPL-3.0.
