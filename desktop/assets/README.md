# Desktop Assets

这个目录专门放桌面版封装相关的静态资源。

当前约定：

- `app.ico`
  Windows 打包时使用的应用图标
- `splash.html`
  桌面版真实启动页与错误展示页

说明：

- `desktop/desktop_app.spec` 会优先读取这里的 `app.ico`
- 如果没有 `app.ico`，打包仍可继续，只是没有自定义图标
- `desktop/desktop_app.py` 会先加载 `splash.html`，等本地服务 ready 后再切换到正式页面
- 后续如果要加托盘图标、关于页图片、安装包资源，也建议统一放在这里
