document.addEventListener('DOMContentLoaded', function() {
    // Common elements
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const typingIndicator = document.getElementById('typing-indicator');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    
    // Login/Signup tab switching logic
    const loginText = document.querySelector(".title-text .login");
    const loginForm = document.querySelector("form.login");
    const loginBtn = document.querySelector("label.login");
    const signupBtn = document.querySelector("label.signup");
    const signupLink = document.querySelector("form .signup-link a");
    
    // Initialize modals if Bootstrap is loaded
    let loginPromptModal = null;
    let clearChatModal = null;
    
    if (typeof bootstrap !== 'undefined') {
        if (document.getElementById('loginPromptModal')) {
            loginPromptModal = new bootstrap.Modal(document.getElementById('loginPromptModal'));
        }
        if (document.getElementById('clearChatModal')) {
            clearChatModal = new bootstrap.Modal(document.getElementById('clearChatModal'));
        }
    }
    
    // Initialize login/signup form switching if present
    if (signupBtn && loginBtn) {
        signupBtn.onclick = (()=>{
            loginForm.style.marginLeft = "-50%";
            loginText.style.marginLeft = "-50%";
        });
        
        loginBtn.onclick = (()=>{
            loginForm.style.marginLeft = "0%";
            loginText.style.marginLeft = "0%";
        });
        
        if (signupLink) {
            signupLink.onclick = (()=>{
                signupBtn.click();
                return false;
            });
        }
    }
    
    // Chat functionality
    if (chatForm && userInput && chatMessages) {
        // Scroll to bottom of chat messages
        function scrollToBottom() {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Add message to chat
        function addMessage(content, isUser = true) {
            // Create message elements
            const messageContainer = document.createElement('div');
            messageContainer.classList.add('message-container');
            messageContainer.classList.add(isUser ? 'user-message' : 'bot-message');
            
            const message = document.createElement('div');
            message.classList.add('message');
            message.innerHTML = content.replace(/\n/g, '<br>');
            
            const messageInfo = document.createElement('div');
            messageInfo.classList.add('message-info');
            
            const messageTime = document.createElement('span');
            messageTime.classList.add('message-time');
            messageTime.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            const messageSender = document.createElement('span');
            messageSender.classList.add('message-sender');
            messageSender.textContent = isUser ? 'You' : 'GAKR';
            
            // Append elements
            if (isUser) {
                messageInfo.appendChild(messageTime);
                messageInfo.appendChild(messageSender);
            } else {
                messageInfo.appendChild(messageSender);
                messageInfo.appendChild(messageTime);
            }
            
            messageContainer.appendChild(message);
            messageContainer.appendChild(messageInfo);
            
            // Clear empty chat message if present
            const emptyChat = chatMessages.querySelector('.empty-chat');
            if (emptyChat) {
                emptyChat.remove();
            }
            
            // Add to DOM and scroll
            chatMessages.appendChild(messageContainer);
            scrollToBottom();
            
            return messageContainer;
        }
        
        // Show typing indicator
        function showTypingIndicator() {
            typingIndicator.classList.remove('d-none');
            scrollToBottom();
        }
        
        // Hide typing indicator
        function hideTypingIndicator() {
            typingIndicator.classList.add('d-none');
        }
        
        // Send message to API
        async function sendMessage(message) {
            try {
                const response = await fetch('/api/chat/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCookie('csrftoken')
                    },
                    body: JSON.stringify({
                        message: message,
                        conversation_id: window.chatConfig ? chatConfig.conversationId : null
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Update conversation ID if provided
                if (data.conversation_id && window.chatConfig) {
                    chatConfig.conversationId = data.conversation_id;
                }
                
                // Check if login required (guest limit reached)
                if (data.require_login && loginPromptModal) {
                    loginPromptModal.show();
                }
                
                // Update guest count if applicable
                if (data.guest_count !== undefined && window.chatConfig) {
                    chatConfig.guestLimit = data.guest_count;
                    
                    // Show warning when approaching limit
                    if (chatConfig.guestLimit >= chatConfig.maxGuestLimit - 1) {
                        const warningElement = document.querySelector('.guest-warning');
                        if (warningElement) {
                            warningElement.classList.remove('d-none');
                        }
                    }
                }
                
                return data.response;
            } catch (error) {
                console.error('Error sending message:', error);
                return `I'm sorry, I encountered an error processing your request: ${error.message}`;
            }
        }
        
        // Handle form submission
        chatForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const message = userInput.value.trim();
            if (!message) return;
            
            // Clear input
            userInput.value = '';
            
            // Add user message to chat
            addMessage(message, true);
            
            // Show typing indicator
            showTypingIndicator();
            
            // Send to API and get response
            const response = await sendMessage(message);
            
            // Hide typing indicator
            hideTypingIndicator();
            
            // Add bot response to chat
            addMessage(response, false);
        });
        
        // Focus input on page load
        userInput.focus();
        
        // Initially scroll to bottom
        scrollToBottom();
        
        // Handle clear chat button
        if (clearChatBtn && clearChatModal) {
            clearChatBtn.addEventListener('click', function() {
                clearChatModal.show();
            });
            
            // Handle confirm clear button
            const confirmClearBtn = document.getElementById('confirm-clear-btn');
            if (confirmClearBtn) {
                confirmClearBtn.addEventListener('click', async function() {
                    try {
                        const response = await fetch('/api/clear-history/', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-CSRFToken': getCookie('csrftoken')
                            },
                            body: JSON.stringify({
                                conversation_id: window.chatConfig ? chatConfig.conversationId : null
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (data.error) {
                            throw new Error(data.error);
                        }
                        
                        // Reload page to show empty state
                        window.location.reload();
                        
                    } catch (error) {
                        console.error('Error clearing chat:', error);
                        alert('Failed to clear conversation. Please try again.');
                    }
                    
                    clearChatModal.hide();
                });
            }
        }
    }
    
    // Helper function to get CSRF token
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});
