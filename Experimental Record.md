# Experimental Record

### Benchmark MATH

|   Mode   | Agents | Steps | Start time |  End time  | Duration |       mask       |
| :------: | :----: | :---: | :--------: | :--------: | :-------: | :--------------: |
| baseline |   2   | 8000 |            |            |          |                  |
| llmshap |   2   | 10000 | 4.10 20:35 | 4.19 10:00 | 205 hours |     1 layer     |
| llmshap |   2   | 10000 | 4.16 17:00 |  underway  | 280 hours |     3 layers     |
| pureshap |   2   | 10000 | 4.19 19:10 |  underway  | 167 hours |      γ = 0      |
| llmshap |   2   | 10000 | 4.19 19:28 |  underway  | 167 hours | γ = 0, 3 layers |
| baseline |   3   | 8000 |            |            |          |                  |
| llmshap |   3   | 12000 | 4.10 01:40 |  stopped  | 640 hours |     1 layer     |
|  masked  |   3   | 12000 |    ---    | 4.14 00:00 |    ---    |     1 layer     |
| baseline |   4   |      |            |            |          |                  |
|  masked  |   4   |      |            |            |          |                  |

### Benchmark camth

|   Mode   | Agents | Steps | Start time |  End time  |  Duration  |     mask     |
| :------: | :----: | :---: | :--------: | :--------: | :--------: | :-----------: |
| baseline |   2   |      |            |            |            |              |
|  masked  |   2   | 20000 | 4.07 15:53 | 4.09 14:00 | 46 hours |   3 layers   |
| llmshap |   2   | 20000 | 4.07 16:13 | 4.13 16:18 | 144 hours | 3 layers(old) |
| llmshap |   2   | 14200 | 4.14 03:25 | 4.19 19:28 | 136 hours |   1 layers   |
| llmshap |   2   | 20000 |            |            |            |   3 layers   |
| baseline |   3   | 20000 |            |            |            |              |
|  masked  |   3   | 20000 | 4.06 15:20 | 4.09 11:20 |  69 hours  |   3 layers   |
| baseline |   4   | 4000 |            |            |            |              |
|  masked  |   4   |      |            |            |            |              |

### Benchmark Codeforces

|  name  | Number of agents | Checkpoint step | Start time | End time | Remark |
| :-----: | :--------------: | :-------------: | :--------: | -------- | :----: |
| llmshap |        2        |                |            |          |        |
| llmshap |        3        |                |            |          |        |
| llmshap |        4        |                |            |          |        |
| llmshap |        2        |                |            |          |        |
