# Сірко

*A chess engine written from scratch in Rust: bitboards, PVS search with Lazy SMP, its own NNUE trained on its own self-play with its own trainer. No code from other engines, no dependencies. Plays on Lichess as [SirkoAI](https://lichess.org/@/SirkoAI). English summary at the end.*

Шаховий рушій, написаний з нуля на Rust, без жодного чужого рушійного коду. Грає на Lichess як [SirkoAI](https://lichess.org/@/SirkoAI). Назва – на честь кошового Івана Сірка, найвідомішого характерника.

## Що всередині

- **Дошка** – бітборди з власними магічними числами, які рушій шукає сам на старті. Хід – 16 біт, позиція копіюється на кожен хід. Класичні шахи й шахи Фішера працюють однаково: рокіровка записується як «король бере свою туру».
- **Пошук** – ітеративне поглиблення з вікнами прагнення, PVS і спільна таблиця переставлень на кількох потоках (Lazy SMP) без блокувань. Також нульовий хід, скорочення пізніх ходів, відсічення марних ходів, однинні продовження, тихий пошук із SEE, ходи-вбивці, історія ходів і контрходи.
- **Оцінка** – власні ваги для середини гри й закінчення: матеріал, положення фігур, рухливість, пішакова структура, прохідні пішаки, безпека короля, загрози й нічийні закінчення.
- **UCI** – обдумування на час суперника (ponder і ponderhit), шахи Фішера, Hash, Threads і Move Overhead. Цього досить для lichess-bot.

## Мережа

Самогра (`sirko datagen ПАРТІЙ ВУЗЛІВ ПОТОКІВ ФАЙЛ`) пише спокійні позиції з оцінкою пошуку й результатом партії. Власний тренер (`sirko train ВИХІД ПРИХОВАНИХ ЕПОХ ДАНІ...`) вчить мережу (768 → H)×2 → 1 зі SCReLU і окремим виходом залежно від кількості фігур, а рушій оцінює нею позиції з покроковим оновленням накопичувачів. Мережа з `nets/sirko.nnue` вбудовується при збиранні, а `setoption name EvalFile` дає перевірити іншу без перезбирання.

## Перевірки

```bash
cargo test --release                                       # perft на еталонних позиціях, SEE, таблиця, симетрія оцінки
./target/release/sirko perft 6                             # 119060324
python3 tools/movegen_check.py ./target/release/sirko      # генерація ходів проти python-chess
python3 tools/uci_check.py ./target/release/sirko          # мати, ponder, шахи Фішера, партія сам із собою
python3 tools/match.py --a ./target/release/sirko --b stockfish --b-opt UCI_LimitStrength=true --b-opt UCI_Elo=2800 --games 30 --tc 8+0.08
```

## Сила

Проти Stockfish з обмеженою силою, контроль 8+0,08 с, один потік:

| Суперник | Результат |
|---|---|
| Stockfish 17.1, Elo 2200 (версія 0.1, ручна оцінка) | 20 з 20 |
| Stockfish 17.1, Elo 2800 (версія 0.1, ручна оцінка) | 14,5 з 30 (+10 =9 −11) |
| Stockfish 19, Elo 3190 (найвища, яку він дозволяє) | 45,7% очок, +305 =487 −408 (1200 партій) |

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

Мережа вбудована в рушій, ручна оцінка лишилася запасною.

## Бот SirkoAI

Сірко грає на Lichess окремим ботом [SirkoAI](https://lichess.org/@/SirkoAI) на тому ж сервері, що й Характерник, і бере його lichess-bot. Грає лише сам рушій: ні баз дебютів, ні таблиць закінчень, ні хмарного аналізу. Коли грають обидва боти, процесор дістається головно Характернику.

Встановлення на сервері з Характерником:

1. Створіть ключ сервера для цього репозиторію й покажіть його:

```bash
sudo install -d -m 755 /etc/sirko && sudo ssh-keygen -q -t ed25519 -N "" -C sirko-server -f /etc/sirko/deploy_key && sudo cat /etc/sirko/deploy_key.pub
```

2. Додайте рядок на GitHub: репозиторій → Settings → Deploy keys → Add deploy key, без дозволу на запис.
3. Завантажте код і встановіть (спитає ключ Lichess акаунта SirkoAI з дозволом «Play games with the bot API»):

```bash
sudo GIT_SSH_COMMAND="ssh -i /etc/sirko/deploy_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" git clone git@github.com:stepovyk/sirko.git /opt/sirko/src && sudo /opt/sirko/src/deploy/sirko-bot install
```

Далі сервер щоночі сам ставить нову версію з гілки `release`, куди CI переносить лише перевірені коміти. Керування: `sirko-bot help`.

## English

Sirko is a UCI chess engine written from scratch in Rust with no dependencies and no code taken from other engines.

- **Board**: bitboards with magic numbers found at startup, copy-make, standard chess and Chess960 (castling encoded as king-takes-rook).
- **Search**: iterative deepening with aspiration windows, PVS, lock-free shared transposition table (Lazy SMP), null move, LMR, futility and late-move pruning, singular extensions, SEE-pruned quiescence, killers, history, countermoves.
- **Evaluation**: a hand-written tapered evaluation and its own NNUE ((768 → H)×2 → 1 with output buckets by piece count, SCReLU, int16, incremental accumulators), trained by its own trainer on its own self-play.
- **Verification**: perft and a differential move generator test against python-chess on tens of thousands of positions, UCI protocol checks, CI on x86 and ARM.

Build: `cargo build --release`. License: GPL-3.0.
