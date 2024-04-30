# SAC 学习笔记

本文既是SAC的阅读笔记，同时也是对PPO和DDPG算法的简要总结，所以先花一定篇幅介绍SAC提出的背景，不感兴趣的读者可以直接跳到第二部分。
## 1. SAC's background

SAC提出前，主流的深度强化学习算法（Deep Reinforcement Learning, DRL）在连续动作空间(continuous action space)的控制任务中已经获得了显著的成果，但各自存在一些缺陷。下面先介绍DeepMind和OpenAI在连续控制领域的经典成果。

### 1.1. DDPG(Deep Deterministic Policy Gradient)

DDPG是基于DPG([Deterministic Policy Gradient](http://proceedings.mlr.press/v32/silver14.pdf))实现的DRL算法。

DPG针对连续动作空间的控制任务在传统的PG（[Policy Gradient](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)，[OpenAI的PG教程](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)）算法上做了改进，将策略函数的输出从一个分布（常用高斯分布）转变为一个唯一确定的动作（通常由一个向量表示，这也是“deterministic“的由来）:

$$ \alpha \sim \pi(\bullet|s) \rightarrow\rightarrow \alpha = \mu(s) $$

同时，DPG引入了AC([Actor-Critic](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f))框架，让值函数（critic）直接指导策略（actor）优化。DDPG可以视为DPG算法的深度学习版实现，并在DPG上加入了几个提高效率的小技巧：Replay buffers, Target networks。

DPG的思路：

以往的PG算法思路是建立累计收益(Cumulative Return)与策略的关系函数，随后调整策略以追求更大的收益。而DPG算法在根本上不同，DPG算法可以被视为 Q-learning 的连续动作空间版本，其思想在于直接利用critic(Q函数)找到可能的最优决策，随后用找到的最优决策来优化策略函数(actor)，也就是说策略调整完全依赖于critic而不用理会实际的收益。

下面我们先定义AC框架的两个基本成分：

#### 1.1.1 Critic （值函数）：
$$ Q^{\mu}_{\omega}(s,a) : S \times A \rightarrow \mathbb{R} $$

$Q$ 函数由参数 （DRL中就是网络参数）控制，将一个状态和动作对映射到一个实数。同 Q-learning，该实数表示，智能体（agent）在状态 执行动作 ，并在之后按照策略 行动，agent 所能获得的预期收益：

$$ Q^{\mu}(s,a) = \mathbb{E} [\sum_{t = 0}^{\infty} \gamma^{t}r(s_{t},a_{t})|s_{0} = s,a_{0} = a] $$

$r(s,a)$ 表示在状态 $s$ 执行动作 $a$ 后，环境反馈的奖励; $\gamma$ 为折扣因子, $\gamma \in (0,1)$ ; $\rho_{\mu}$ 则代表agent按照策略 $\mu$ 行动时，它会遇到的动作、状态对所服从的分布

#### 1.1.2 Actor（策略函数）：
$$ \mu_{\theta}(s): S \rightarrow A $$

策略函数由参数 $\theta$ 确定，将状态空间的一个点映射到动作空间的一个点，这也就是决策过程。算法公式：定义了DPG需要的两个主要成分后，利用critic直接优化actor的思想，就有了DPG的核心公式：

$$ \max_{\theta}\mathbb{E}_{s \sim Data} [Q_{\omega}^{\mu}(s,\mu_{\theta}(s))] \tag{1.1} $$ 

此时，$Q$ 函数参数固定，只调整策略函数 $\mu$ ；$Q$ 函数另外单独训练，训练方式同 Q-learning(须额外描述)。


寻找最优决策：DPG的最终目的还是在于调整策略，所以先假设已有最优的critic: $Q^{*}$ ，在 $t$ 时刻遇到某个状态 $s_{t}$ ，要求此刻最优动作 $a^{*}_{t}$ , 我们需要做的就是固定住 $Q^{*}$ 函数输入端 $(s,a)$ 当中的状态 $s$ 为 $s_{t}$ ,不断调整输入动作 $a$ ，直到 $Q^{*}$ 的输出值最大，此时的动作 $a$ 即为最优决策，这个操作等价于一般的Q-learning中的greedy决策方式：

$$ a_{t}^{*} = argmax_{a} Q^{*}(s_{t},a) \tag{1.2} $$ 

形象来说，假如我们面临一个需要做决策的情况，而我们知道所有决策的后果（ 

函数会告诉我们），那么最优决策就是那个可以带来最佳后果的决策。不过不同于Q-learning，在DPG的目标任务中，动作空间是连续的，所以可以直接让 函数对输入动作

    求导，这就成了一个将指定的连续可导函数最大化的问题。

优化当前策略：在动作空间中找到针对状态
最优动作 后，我们就可以根据这个最优动作调整策略，即优化actor: 。让actor模仿这一决策即可（可以把这一步视为监督学习，最优动作就是其标签，优化过程就是让actor在接受

时的输出向标签靠拢）：

不过这次策略调整只改变了智能体（agent）在
这单个状态下的决策，接下来还需要将整个状态空间 中所有的状态的策略调整到最优，就可以得到最优策略

。

整合成DPG：以上我们将DPG拆成了两步：1.找最优决策；2.优化策略。

然而训练actor时，critic接收的输入动作就是actor的输出(

)，所以两步可以直接简化成:

在实际的算法实现中这一优化过程不是向上面一样一步完成的，毕竟一开始学习时，并没有最优critic（
），深度学习也要求网络参数的更新不能太快。在训练时，actor与环境交互所得的数据会用于训练critic，使之更加准确，向最优 靠拢；actor也会根据当前的critic调整自己输出的动作，向最优策略

靠近。

PS:

    根据上述可知，DPG及DDPG要求Q函数对动作求导，所以动作空间必然连续，这将导致DPG算法不适用于离散动作空间的任务。
    训练策略函数时，对于策略的调整需要 

函数对动作的导数，所以必然使用AC框架，同时对策略的调整是好是坏完全取决于critic是否准确，最终策略成为

    函数的附属品，这也导致整个DPG算法其实更偏向于value-based算法（尽管名为policy gradient）。然而，这也给DPG算法带来一个优势——更好地利用过往数据（更加off-policy，这是由于Q-learning本身就是off-policy算法，critic自然可以更多利用过往的数据）。

衍生成果：

D4PG（引入分布式的critic，并使用多个actor（learner）共同与环境交互）

TD3（参考了double Q-learning的思想来优化critic，延缓actor的更新，计算critic的优化目标时在action上加一个小扰动）
1.2. PPO(Proximal Policy Optimization Algorithms)

PPO是TRPO(Trust Region Policy Optimization)的简化版，二者的目标都是：在PG算法的优化过程中，使性能单调上升，并且使上升的幅度尽量大。

PPO同样使用了AC框架，不过相比DPG更加接近传统的PG算法，采用的是随机分布式的策略函数（Stochastic Policy），智能体（agent）每次决策时都要从策略函数输出的分布中采样，得到的样本作为最终执行的动作，因此天生具备探索环境的能力，不需要为了探索环境给决策加上扰动；PPO的重心会放到actor上，仅仅将critic当做一个预测状态好坏（在该状态获得的期望收益）的工具，策略的调整基准在于获取的收益，不是critic的导数。

PPO的思路：

PPO的基本思想跟PG算法一致，便是直接根据策略的收益好坏来调整策略。

作为一个AC算法，它的基本框架跟Stochastic Actor-critic算法一致，所以先定义PPO的策略函数actor：

此时动作
服从一个受参数 控制的条件分布，可以理解为，假如面对某个状态 ，agent做出决策动作 的概率

。

再定义一个值函数critic：

将状态映射到实数，该实数表示agent在状态
会获得的期望收益， 是函数参数。与 函数稍有不同， 函数不考虑某个具体的决策的后果，而是综合考虑agent的行动满足策略 时，获得收益的期望：

。

它与DPG中提到的

函数存在联系：

表示 状态的下一步状态。 函数与 函数本质上是值函数的两种表达方式，并且二者可以相互转化。而PPO中不需要求 函数对动作的导数，所以使用 函数即可，另外

    函数的输入少了动作，输入空间小了很多，大大简化值函数。

定义好actor与critic后，开始引出PPO的算法：

(1) Stochastic Actor-critic的单步策略优化的目标函数可以表示为：

其中，
，这是advantage项，表示在 状态下某一决策 带来的期望收益( )相比原本的期望收益(

)高多少（或者差多少），如果与原本决策一致则advantage为零。

    原本 

这个位置是决策的期望收益即 ，不过用advantage代替期望收益可以增加算法的稳定性，不过无论用哪个，公式的目的都是通过调整策略获取更高的期望收益。可以理解为，如果 值为正，则提高

    ，以增加相应决策出现的概率；反之则降低概率。

(2) 加上importance sampling技术，使其成为off-policy算法：

表示采集数据时与环境交互的策略， 表示当前正在训练的策略，此时因为训练的策略与采集数据时的策略不同，引入 项进行修正，使其接近on-policy（式1.5）的效果。由于训练策略

    时的数据不是源于它自己与环境的交互，因此这被称为off-policy算法。

(3) 而为了实现（修正：近似地的接近TRPO，不能保证单调提升）性能的单调上升，PPO的做法是训练策略时，强行限制住策略的更新速度，

其中
是一个需要手动调整的参数，大于0。在clip函数的帮助下， 的值被限制在 中，确定了策略优化时的变化幅度不会太大。