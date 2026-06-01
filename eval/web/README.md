# RULER 结果可视化

本目录提供一个无外部依赖的本地网页，用于对比 `eval/exp_analysis/ruler_results` 下的 RULER 汇总结果。

## 启动

```bash
python "/zouxiangyu/codes/Learning/In-Place-TTT/eval/web/server.py" --host "0.0.0.0" --port 8765
```

打开：

```text
http://127.0.0.1:8765/
```

## 数据来源

服务端会在每次请求 `/api/ruler-results` 时扫描结果目录下的：

```text
*/summary_all_lengths.json
```

同时兼容用户手动生成的旧拼写：

```text
*/summary_all_lenghts.json
```

新增模型结果后不需要改前端代码，点击页面上的“刷新数据”即可重新读取；开启“自动刷新”后会每 30 秒重新扫描一次。

默认视图只展示 Base 结果：无 chat template，且排除 smoke。也可以在页面左侧切换到全部结果、chat template 或 smoke。
