function sendMessage() {
    var userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") return;

    addUserBubble(userInput);
    fetchResponse(userInput);
    document.getElementById("user-input").value = "";
}

function addUserBubble(message) {
    var chatWindow = document.getElementById("chat-window");
    var bubble = document.createElement("div");
    bubble.className = "bubble user-bubble";
    bubble.textContent = message;
    chatWindow.appendChild(bubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addBotBubble(message) {
    var chatWindow = document.getElementById("chat-window");
    var bubble = document.createElement("div");
    bubble.className = "bubble bot-bubble";
    bubble.textContent = message;
    chatWindow.appendChild(bubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function fetchResponse(userInput) {
    // 在这里调用后端接口获取回复
    // 这里是一个示例，假设后端接口直接返回回复消息
    var botResponse = "这是一个示例回复。";
    setTimeout(function() {
        addBotBubble(botResponse);
    }, 1000); // 模拟延迟，以便更真实地模拟异步请求
}
