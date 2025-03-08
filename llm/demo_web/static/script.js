let questionIndex = 0;  // Start from the first question

window.onload = function() {
    loadNextQuestion();
    // Add event listener for enter key on input
    document.getElementById("userInput").addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            event.preventDefault(); // Prevent the default action to stop form submission
            sendAnswer(); // Call sendAnswer function
        }
    });
};

function loadNextQuestion() {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/get_question", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            if (response.question) {
                addMessage(response.question, 'bot');
            } else {
                addMessage("问卷结束，谢谢参与！", 'bot');
                document.querySelector(".input-container").style.display = 'none';
            }
        }
    };
    xhr.send(JSON.stringify({questionIndex: questionIndex}));
}

function sendAnswer() {
    var userInput = document.getElementById("userInput").value;
    if (userInput.trim() === '') {
        alert('Please type an answer.');
        return;
    }
    addMessage(userInput, 'user');
    document.getElementById("userInput").value = '';  // Clear input field

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/process_text", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
            var data = JSON.parse(xhr.responseText);
            addMessage(data.reply, 'bot');
            questionIndex++;
            loadNextQuestion();  // Load the next question
        }
    };
    xhr.send(JSON.stringify({answer: userInput, questionIndex: questionIndex}));
}

function addMessage(message, sender) {
    var chatBox = document.getElementById("chatBox");
    var messageElement = document.createElement('div');
    messageElement.className = 'message ' + sender;
    messageElement.innerText = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;  // Auto-scroll to the latest message
}
