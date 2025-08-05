# Pluribus Poker AI 中文使用说明

本项目是一个基于 Python 的扑克 AI 框架，专注于实现类似 Pluribus 的多人非限制下注德州扑克 AI。你可以用它来训练 AI 模型、与 AI 对战，或者将其核心逻辑集成到你自己的项目中。

## 快速开始

你可以通过项目的命令行界面来启动一个游戏。

```bash
# 后面我们会探讨如何修改玩家人数
python -m poker_ai.cli.runner
```

## 重点问题解答

### 1. 如何在我的 Python 项目中调用这个 AI？

本项目被设计成一个标准的 Python 包，因此可以很方便地在其他项目中使用。

**步骤 1：安装**

在你的虚拟环境中，进入 `pluribus-poker-AI` 项目的根目录，然后使用 pip 进行本地安装：

```bash
pip install .
```

这会将 `poker_ai` 安装到你的 Python 环境中，就像安装其他库（如 `requests` 或 `numpy`）一样。

**步骤 2：在你的代码中调用**

安装后，你就可以在任何 Python 文件中导入并使用它的模块了。例如，如果你想使用游戏引擎：

```python
from poker_ai.poker.engine import Engine
from poker_ai.poker.player import Player

# 示例：创建一个游戏引擎并开始游戏
# (具体参数需要参考 Engine 类的实现)
players = [Player(stack=1000) for _ in range(6)] # 创建6个玩家
engine = Engine(players)
engine.start_game()
```

### 2. 能否设置为6人桌游戏？

**可以。**

通过分析代码结构，游戏的玩家数量很可能是在创建 `Engine`（游戏引擎）或通过命令行启动时配置的。

*   **通过代码修改**：
    在创建 `Engine` 实例时，传递一个包含6个 `Player` 对象的列表即可，如上一节的示例代码所示。

*   **通过命令行修改**：
    项目的命令行接口 (`poker_ai/cli/runner.py`) 很可能支持通过参数来指定玩家数量。你需要查看该文件的具体实现，但通常会类似这样：
    ```bash
    # 这是一个推测的命令，具体参数名可能不同
    python -m poker_ai.cli.runner --num-players 6
    ```
    如果想确认，可以运行 `python -m poker_ai.cli.runner --help` 查看所有可用的命令行选项。

### 3. 是否有"输入游戏历史记录，提供决策"的方法？

项目**可能没有**一个直接接收“人类可读的游戏历史文本”并返回决策的简单函数。AI 的工作方式是基于一个精确的、机器可读的**游戏状态 (Game State)**。

不过，你可以通过模拟这个过程来获得 AI 的决策，这也是集成 AI 的核心方式。

### 4. 如何模拟获取 AI 的决策？

你需要以程序化的方式，一步步构建出你想要 AI 进行决策时的那个游戏局面 (`GameState` 对象)，然后将这个状态交给 AI 的 `agent` (智能体) 来获取决策。

**基本流程如下：**

1.  **加载 AI Agent**: 首先，你需要加载一个训练好的 AI 模型。根据文件列表，模型可能保存在 `.joblib` 文件中。
    ```python
    import joblib
    from poker_ai.ai.agent import Agent # 假设 Agent 类在这里

    # 加载预训练的 AI 模型
    ai_agent = joblib.load('path/to/your/agent.joblib')
    ```

2.  **构建游戏状态 (GameState)**: 这是最关键的一步。你需要手动创建一个 `GameState` 对象，并填充所有必要的信息，以完整描述当前的牌局。你需要参考 `poker_ai/poker/state.py` 文件来了解 `GameState` 类的具体结构。它通常包含：
    *   当前玩家的手牌 (`hand`)
    *   公共牌 (`board`)
    *   底池大小 (`pot_size`)
    *   每个玩家的下注情况 (`bets`)
    *   当前轮次的行动历史 (`history`)
    *   当前轮到谁做决策 (`current_player_index`)
    *   等等...

    ```python
    from poker_ai.poker.state import GameState
    from poker_ai.poker.card import Card

    # 这是一个非常简化的示例，具体属性需要完全匹配 GameState 类的定义
    current_game_state = GameState(
        hands=[[Card('As'), Card('Ks')], [Card('Qd'), Card('Jd')]], # 假设这是两个玩家的手牌
        board=[Card('Ah'), Card('Kd'), Card('2c')], # 翻牌圈的公共牌
        street=1, # 0: preflop, 1: flop, 2: turn, 3: river
        bets=[10, 20], # 玩家的下注额
        pot=30,
        current_player_idx=1 # 轮到第二个玩家决策
    )
    ```

3.  **获取决策**: 一旦你构建了精确的 `GameState` 对象，就可以调用 AI Agent 的决策方法来获取一个合法的动作（Action）。
    ```python
    # Agent 类中应该有一个类似 get_action 的方法
    action = ai_agent.get_action(current_game_state)

    print(f"AI 的决策是: {action}") # action 可能是一个枚举类型或字符串，如 FOLD, CALL, RAISE
    ```

通过以上三个步骤，你就可以在任何需要的时候，为任何你构建的牌局局面，获取来自这个 Poker AI 的决策建议。这是将 AI 集成到外部应用（例如，一个带有图形界面的扑克游戏或一个复盘工具）的标准方法。
