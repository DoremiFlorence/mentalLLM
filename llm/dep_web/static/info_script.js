document.addEventListener('DOMContentLoaded', () => {
    const infoSubmitButton = document.getElementById('info-submit-button');

    infoSubmitButton.addEventListener('click', () => {
        const nickname = document.getElementById('nickname').value;
        const age = document.getElementById('age').value;
        const job = document.getElementById('job').value;

        if (!nickname || !age || !job) {
            alert('请填写所有信息');
            return;
        }

        // 将用户信息存储在 localStorage 中
        localStorage.setItem('nickname', nickname);
        localStorage.setItem('age', age);
        localStorage.setItem('job', job);

        // 跳转到聊天界面
        window.location.href = '/chat';
    });
});
