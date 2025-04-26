// GAKR AI Chat Interface

// Initialize variables once DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    const chatInput = document.getElementById('chat-input');
    const chatOutput = document.getElementById('chat-output');
    const sendButton = document.getElementById('send-button');
    const chatForm = document.getElementById('chat-form');

    let conversationHistory = []; // To store the conversation on the client-side temporarily

    function appendMessage(sender, message, isNew = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender === 'User' ? 'message-user' : 'message-ai'}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = message;
        
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-timestamp small text-muted mt-1';
        
        const now = new Date();
        const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        metaDiv.innerHTML = `<span class="sender">${sender}</span> • ${timeString}`;
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(metaDiv);
        
        chatOutput.appendChild(messageDiv);
        chatOutput.scrollTop = chatOutput.scrollHeight; // Scroll to the latest message
        
        // Add to conversation history if it's a new message
        if (isNew) {
            conversationHistory.push({ 
                sender: sender === 'User' ? 'user' : 'bot', 
                message: message 
            });
        }
    }

    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message message-ai typing-indicator';
        typingDiv.id = 'typing-indicator';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `
            <div class="d-flex align-items-center">
                <div class="spinner-grow spinner-grow-sm text-primary me-2" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <span>GAKR AI is thinking...</span>
            </div>
        `;
        
        typingDiv.appendChild(contentDiv);
        chatOutput.appendChild(typingDiv);
        chatOutput.scrollTop = chatOutput.scrollHeight;
    }

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    async function sendMessage(userMessage) {
        if (!userMessage.trim()) return;
        
        // Display user message
        appendMessage('User', userMessage);
        
        // Clear input field
        chatInput.value = '';
        
        // Show typing indicator
        showTypingIndicator();
        
        try {
            // Send message to server
            const response = await fetch('/api/gakr', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken() // Get CSRF token from cookies
                },
                body: JSON.stringify({ 
                    message: userMessage, 
                    history: conversationHistory
                })
            });

            // Remove typing indicator
            removeTypingIndicator();
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const botResponse = data.response;
            
            // Display bot response
            appendMessage('GAKR AI', botResponse);
            
        } catch (error) {
            console.error('Error sending message:', error);
            removeTypingIndicator();
            appendMessage('Error', 'Failed to get response. Please try again.');
        }
    }

    // Function to get CSRF token from cookies
    function getCsrfToken() {
        const name = 'csrf_token=';
        const decodedCookie = decodeURIComponent(document.cookie);
        const cookieArray = decodedCookie.split(';');
        
        for (let i = 0; i < cookieArray.length; i++) {
            let cookie = cookieArray[i].trim();
            if (cookie.indexOf(name) === 0) {
                return cookie.substring(name.length, cookie.length);
            }
        }
        return '';
    }

    // Add event listener for form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const userMessage = chatInput.value.trim();
        sendMessage(userMessage);
    });

    // Add event listener for Enter key in input field
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const userMessage = chatInput.value.trim();
            sendMessage(userMessage);
        }
    });

    // Optional: Load previous history when the page loads
    // This can be enabled if using client-side storage
    function loadPreviousHistory() {
        const messageElements = chatOutput.querySelectorAll('.message');
        
        messageElements.forEach(element => {
            const isBotMessage = element.classList.contains('message-ai');
            const isUserMessage = element.classList.contains('message-user');
            
            if (isBotMessage || isUserMessage) {
                const contentElement = element.querySelector('.message-content');
                const message = contentElement.textContent;
                const sender = isBotMessage ? 'bot' : 'user';
                
                conversationHistory.push({ sender, message });
            }
        });
    }

    // Initialize by loading any existing messages in the DOM
    loadPreviousHistory();
});