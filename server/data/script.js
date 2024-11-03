// Check for browser support
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    const startButton = document.getElementById('start-btn');
    const stopButton = document.getElementById('stop-btn');
    const resultDiv = document.getElementById('result');

    recognition.continuous = false; // Stop automatically after detecting speech
    recognition.interimResults = true; // Show interim results

    // Start listening
    startButton.addEventListener('click', () => {
        recognition.start();
        startButton.disabled = true;
        stopButton.disabled = false;
    });

    // Stop listening
    stopButton.addEventListener('click', () => {
        recognition.stop();
        startButton.disabled = false;
        stopButton.disabled = true;
    });

    // Capture results
    recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
            .map(result => result[0].transcript)
            .join('');

        resultDiv.textContent = transcript;
    };

    // Handle errors
    recognition.onerror = (event) => {
        console.error('Error occurred in recognition: ' + event.error);
    };

    // Restart the buttons when recognition ends
    recognition.onend = () => {
        startButton.disabled = false;
        stopButton.disabled = true;
    };
} else {
    alert('Speech recognition not supported in this browser.');
}
