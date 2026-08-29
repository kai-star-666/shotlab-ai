// 部署完整 Python 分析服务后，只需修改这一处地址。
const ANALYZER_URL = "https://share.streamlit.io/deploy?owner=kai-star-666&repo=shotlab-ai&branch=main&mainModule=app.py";

for (const link of document.querySelectorAll(".js-analyzer-link")) {
  link.href = ANALYZER_URL;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
}

if (/MicroMessenger/i.test(navigator.userAgent)) {
  document.documentElement.classList.add("wechat-browser");
}
