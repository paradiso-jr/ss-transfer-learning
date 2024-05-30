# EXP logging
0512
    完成了transfer learning部分的代码，但是发现训练效果不好，需要进一步调试。
    目前看来，可能是使用decoder后的embedding来做transfer，而得来的梯度，无法更新encoder的参数。
    需要进一步调试。但这样是否效果真的不好，还没有验证。

0514
    dev branch用来使用最原始的transform来作为base model，然后使用transfer learning的方法来训练。