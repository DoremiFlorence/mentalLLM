let now_ques = "您是否经常因为一些小事而烦恼？";
let total_score = 0;

document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const submitButton = document.getElementById('submit-button');

    const nickname = localStorage.getItem('nickname');

    const appendMessage = (text, sender) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);
        messageElement.innerText = text;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    const sendMessage = () => {
        const userText = userInput.value;
        if (!userText) return;

        appendMessage(userText, 'user');
        userInput.value = '';

        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_input: userText,
                now_ques: now_ques,
                total_score: total_score
            })
        })
        .then(response => response.json())
        .then(data => {
            appendMessage(data.reply, 'bot');
            total_score = data.total_score;
            explain=data.explain
            if (data.finished) {
                appendMessage(`问卷结束～\n最终得分是: ${total_score}。${explain}`, 'bot');

            } else {
                now_ques = data.next_question;
                appendMessage(now_ques, 'bot');
            }
        })
        .catch(error => console.error('Error:', error));
    };

    appendMessage(`欢迎您，${nickname}！我们将开始问卷调查。`, 'bot');
    appendMessage(now_ques, 'bot');

    submitButton.addEventListener('click', sendMessage);

    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});
