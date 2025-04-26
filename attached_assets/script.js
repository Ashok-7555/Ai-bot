/**
 * GAKR AI - Main JavaScript file
 * Contains shared functionality for the GAKR AI application
 */

document.addEventListener('DOMContentLoaded', function() {
    // Tutorial system
    const tutorialHelper = {
        // Store tutorial steps
        steps: [],
        currentStep: 0,
        isActive: false,
        
        // Initialize tutorial
        init: function(steps, onComplete) {
            this.steps = steps;
            this.currentStep = 0;
            this.onComplete = onComplete || function() {};
            this.createTutorialOverlay();
        },
        
        // Create tutorial overlay
        createTutorialOverlay: function() {
            // Remove existing overlay if any
            const existingOverlay = document.getElementById('tutorial-overlay');
            if (existingOverlay) {
                existingOverlay.remove();
            }
            
            // Create new overlay
            const overlay = document.createElement('div');
            overlay.id = 'tutorial-overlay';
            overlay.className = 'tutorial-overlay';
            overlay.innerHTML = `
                <div class="tutorial-highlight"></div>
                <div class="tutorial-tooltip">
                    <div class="tutorial-tooltip-header">
                        <span class="tutorial-step">Step <span class="tutorial-step-number">1</span>/<span class="tutorial-total-steps">${this.steps.length}</span></span>
                        <button class="tutorial-close-button">&times;</button>
                    </div>
                    <div class="tutorial-tooltip-content">
                        <h3 class="tutorial-title"></h3>
                        <p class="tutorial-description"></p>
                    </div>
                    <div class="tutorial-tooltip-footer">
                        <button class="tutorial-prev-button">Previous</button>
                        <button class="tutorial-next-button">Next</button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(overlay);
            
            // Set up event listeners
            overlay.querySelector('.tutorial-close-button').addEventListener('click', () => {
                this.stop();
            });
            
            overlay.querySelector('.tutorial-next-button').addEventListener('click', () => {
                this.next();
            });
            
            overlay.querySelector('.tutorial-prev-button').addEventListener('click', () => {
                this.prev();
            });
        },
        
        // Start the tutorial
        start: function() {
            this.isActive = true;
            this.showStep(0);
            document.getElementById('tutorial-overlay').classList.add('active');
            
            // Track that user has started the tutorial
            this.trackTutorialProgress('started');
        },
        
        // Stop the tutorial
        stop: function() {
            this.isActive = false;
            document.getElementById('tutorial-overlay').classList.remove('active');
            
            // Track that user has stopped the tutorial
            this.trackTutorialProgress('stopped');
        },
        
        // Show specific step
        showStep: function(stepIndex) {
            if (stepIndex < 0 || stepIndex >= this.steps.length) return;
            
            const step = this.steps[stepIndex];
            this.currentStep = stepIndex;
            
            // Update tooltip content
            const overlay = document.getElementById('tutorial-overlay');
            overlay.querySelector('.tutorial-step-number').textContent = stepIndex + 1;
            overlay.querySelector('.tutorial-title').textContent = step.title;
            overlay.querySelector('.tutorial-description').textContent = step.description;
            
            // Update buttons
            overlay.querySelector('.tutorial-prev-button').disabled = stepIndex === 0;
            const nextButton = overlay.querySelector('.tutorial-next-button');
            
            if (stepIndex === this.steps.length - 1) {
                nextButton.textContent = 'Finish';
            } else {
                nextButton.textContent = 'Next';
            }
            
            // Position highlight and tooltip
            this.positionElements(step);
            
            // Track step view
            this.trackTutorialProgress('step', stepIndex + 1);
        },
        
        // Position highlight and tooltip
        positionElements: function(step) {
            const targetElement = document.querySelector(step.element);
            if (!targetElement) return;
            
            const overlay = document.getElementById('tutorial-overlay');
            const highlight = overlay.querySelector('.tutorial-highlight');
            const tooltip = overlay.querySelector('.tutorial-tooltip');
            
            // Position highlight
            const rect = targetElement.getBoundingClientRect();
            highlight.style.top = rect.top + 'px';
            highlight.style.left = rect.left + 'px';
            highlight.style.width = rect.width + 'px';
            highlight.style.height = rect.height + 'px';
            
            // Position tooltip
            const tooltipRect = tooltip.getBoundingClientRect();
            const position = step.position || 'bottom';
            
            switch (position) {
                case 'top':
                    tooltip.style.top = (rect.top - tooltipRect.height - 10) + 'px';
                    tooltip.style.left = (rect.left + (rect.width / 2) - (tooltipRect.width / 2)) + 'px';
                    break;
                case 'bottom':
                    tooltip.style.top = (rect.bottom + 10) + 'px';
                    tooltip.style.left = (rect.left + (rect.width / 2) - (tooltipRect.width / 2)) + 'px';
                    break;
                case 'left':
                    tooltip.style.top = (rect.top + (rect.height / 2) - (tooltipRect.height / 2)) + 'px';
                    tooltip.style.left = (rect.left - tooltipRect.width - 10) + 'px';
                    break;
                case 'right':
                    tooltip.style.top = (rect.top + (rect.height / 2) - (tooltipRect.height / 2)) + 'px';
                    tooltip.style.left = (rect.right + 10) + 'px';
                    break;
            }
            
            // Ensure tooltip stays within viewport
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            const tooltipTop = parseInt(tooltip.style.top);
            const tooltipLeft = parseInt(tooltip.style.left);
            
            if (tooltipLeft < 10) {
                tooltip.style.left = '10px';
            } else if (tooltipLeft + tooltipRect.width > viewportWidth - 10) {
                tooltip.style.left = (viewportWidth - tooltipRect.width - 10) + 'px';
            }
            
            if (tooltipTop < 10) {
                tooltip.style.top = '10px';
            } else if (tooltipTop + tooltipRect.height > viewportHeight - 10) {
                tooltip.style.top = (viewportHeight - tooltipRect.height - 10) + 'px';
            }
        },
        
        // Go to next step
        next: function() {
            if (this.currentStep === this.steps.length - 1) {
                this.complete();
            } else {
                this.showStep(this.currentStep + 1);
            }
        },
        
        // Go to previous step
        prev: function() {
            if (this.currentStep > 0) {
                this.showStep(this.currentStep - 1);
            }
        },
        
        // Complete tutorial
        complete: function() {
            this.stop();
            this.onComplete();
            
            // Track that user has completed the tutorial
            this.trackTutorialProgress('completed');
        },
        
        // Track tutorial progress
        trackTutorialProgress: function(action, stepNumber) {
            // Send tracking data to server
            fetch('/api/onboarding', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    tutorial_action: action,
                    tutorial_step: stepNumber || 0,
                    tutorial_completed: action === 'completed'
                })
            }).catch(error => {
                console.error('Error tracking tutorial progress:', error);
            });
        }
    };
    
    // Chat functionality
    const chatHelper = {
        // Initialize chat
        init: function(containerId, inputId, submitButtonId) {
            this.container = document.getElementById(containerId);
            this.input = document.getElementById(inputId);
            this.submitButton = document.getElementById(submitButtonId);
            
            if (!this.container || !this.input || !this.submitButton) return;
            
            this.setupEventListeners();
        },
        
        // Set up event listeners
        setupEventListeners: function() {
            // Auto-resize textarea as user types
            this.input.addEventListener('input', () => {
                this.input.style.height = 'auto';
                const newHeight = Math.min(120, this.input.scrollHeight);
                this.input.style.height = newHeight + 'px';
                
                // Enable/disable submit button based on input
                if (this.input.value.trim().length > 0) {
                    this.submitButton.classList.remove('disabled');
                } else {
                    this.submitButton.classList.add('disabled');
                }
            });
            
            // Handle Enter key for sending (allow Shift+Enter for new line)
            this.input.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' && !event.shiftKey && !this.submitButton.classList.contains('disabled')) {
                    event.preventDefault();
                    this.sendMessage();
                }
            });
            
            // Handle submit button click
            this.submitButton.addEventListener('click', () => {
                if (!this.submitButton.classList.contains('disabled')) {
                    this.sendMessage();
                }
            });
        },
        
        // Send message
        sendMessage: function() {
            const text = this.input.value.trim();
            if (!text) return;
            
            // Add user message to chat
            this.addMessage(text, 'user');
            
            // Clear input and reset height
            this.input.value = '';
            this.input.style.height = 'auto';
            this.submitButton.classList.add('disabled');
            
            // Show typing indicator
            this.showTypingIndicator();
            
            // Scroll to bottom
            this.scrollToBottom();
            
            // Send to server
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: text })
            })
            .then(response => response.json())
            .then(data => {
                // Hide typing indicator
                this.hideTypingIndicator();
                
                // Process and display response
                let responseText = data;
                if (typeof data === 'object') {
                    responseText = data.response || JSON.stringify(data);
                }
                
                // Add AI message to chat
                this.addMessage(responseText, 'ai');
            })
            .catch(error => {
                console.error('Error:', error);
                this.hideTypingIndicator();
                this.addMessage('Sorry, I encountered an error. Please try again.', 'ai');
            });
        },
        
        // Add message to chat
        addMessage: function(text, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `gemini-message gemini-message-${type}`;
            
            const headerDiv = document.createElement('div');
            headerDiv.className = 'gemini-message-header';
            
            const avatarDiv = document.createElement('div');
            avatarDiv.className = 'gemini-message-avatar';
            
            const avatarIcon = document.createElement('i');
            avatarIcon.className = type === 'user' ? 'fas fa-user' : 'fas fa-robot';
            
            const nameSpan = document.createElement('span');
            nameSpan.textContent = type === 'user' ? 'You' : 'GAKR AI';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'gemini-message-content';
            contentDiv.textContent = text;
            
            avatarDiv.appendChild(avatarIcon);
            headerDiv.appendChild(avatarDiv);
            headerDiv.appendChild(nameSpan);
            
            messageDiv.appendChild(headerDiv);
            messageDiv.appendChild(contentDiv);
            
            this.container.appendChild(messageDiv);
            
            this.scrollToBottom();
        },
        
        // Show typing indicator
        showTypingIndicator: function() {
            // Check if typing indicator already exists
            let typingIndicator = document.getElementById('typingIndicator');
            
            if (!typingIndicator) {
                typingIndicator = document.createElement('div');
                typingIndicator.id = 'typingIndicator';
                typingIndicator.className = 'gemini-typing';
                
                for (let i = 0; i < 3; i++) {
                    const dot = document.createElement('div');
                    dot.className = 'gemini-typing-dot';
                    typingIndicator.appendChild(dot);
                }
                
                this.container.appendChild(typingIndicator);
            } else {
                typingIndicator.classList.remove('d-none');
            }
            
            this.scrollToBottom();
        },
        
        // Hide typing indicator
        hideTypingIndicator: function() {
            const typingIndicator = document.getElementById('typingIndicator');
            if (typingIndicator) {
                typingIndicator.classList.add('d-none');
            }
        },
        
        // Scroll chat to bottom
        scrollToBottom: function() {
            this.container.scrollTop = this.container.scrollHeight;
        }
    };
    
    // Onboarding wizard functionality
    const onboardingHelper = {
        // Initialize onboarding wizard
        init: function() {
            // Set up active slide
            this.activeSlide = 1;
            this.steps = document.querySelectorAll('.step');
            this.slideWrapper = document.querySelector('.slide-wrapper');
            this.slides = document.querySelectorAll('.slide');
            this.nextButtons = document.querySelectorAll('.nav-button-next');
            this.prevButtons = document.querySelectorAll('.nav-button-prev');
            
            this.experienceOptions = document.querySelectorAll('.slide[data-slide="2"] .option-card');
            this.responseOptions = document.querySelectorAll('.response-length');
            this.modelOptions = document.querySelectorAll('.model-card');
            this.tutorialOptions = document.querySelectorAll('.tutorial-option');
            
            this.favoriteTopicsInput = document.getElementById('favorite-topics');
            
            this.tutorialModal = document.getElementById('tutorialModal');
            this.closeTutorial = document.getElementById('closeTutorial');
            this.completeTutorial = document.getElementById('completeTutorial');
            
            // State object to store user selections
            this.userPreferences = {
                experienceLevel: null,
                favoriteTopics: [],
                responseLength: null,
                modelType: null,
                showTutorial: null
            };
            
            this.setupEventListeners();
            this.updateActiveSlide(this.activeSlide);
        },
        
        // Set up event listeners
        setupEventListeners: function() {
            // Set up option selection for experience level
            this.experienceOptions.forEach(option => {
                option.addEventListener('click', () => {
                    this.experienceOptions.forEach(opt => opt.classList.remove('selected'));
                    option.classList.add('selected');
                    this.userPreferences.experienceLevel = option.dataset.value;
                    this.enableNextButton(2);
                });
            });
            
            // Set up option selection for response length
            this.responseOptions.forEach(option => {
                option.addEventListener('click', () => {
                    this.responseOptions.forEach(opt => opt.classList.remove('selected'));
                    option.classList.add('selected');
                    this.userPreferences.responseLength = option.dataset.value;
                });
            });
            
            // Listen to favorite topics input
            if (this.favoriteTopicsInput) {
                this.favoriteTopicsInput.addEventListener('input', () => {
                    if (this.favoriteTopicsInput.value.trim()) {
                        this.userPreferences.favoriteTopics = this.favoriteTopicsInput.value.split(',').map(topic => topic.trim());
                    } else {
                        this.userPreferences.favoriteTopics = [];
                    }
                });
            }
            
            // Set up option selection for model type
            this.modelOptions.forEach(option => {
                option.addEventListener('click', () => {
                    this.modelOptions.forEach(opt => opt.classList.remove('selected'));
                    option.classList.add('selected');
                    this.userPreferences.modelType = option.dataset.value;
                    this.enableNextButton(4);
                });
            });
            
            // Set up option selection for tutorial
            this.tutorialOptions.forEach(option => {
                option.addEventListener('click', () => {
                    this.tutorialOptions.forEach(opt => opt.classList.remove('selected'));
                    option.classList.add('selected');
                    this.userPreferences.showTutorial = option.dataset.value === 'yes';
                    this.enableNextButton(5);
                });
            });
            
            // Add click event listeners to navigation buttons
            this.nextButtons.forEach(button => {
                button.addEventListener('click', () => {
                    if (button.hasAttribute('disabled')) return;
                    
                    const nextSlide = parseInt(button.dataset.next);
                    if (nextSlide > this.activeSlide) {
                        // Handle specific slide transitions
                        if (this.activeSlide === 5 && nextSlide === 6 && this.userPreferences.showTutorial) {
                            this.showTutorial();
                        }
                        
                        // If going to the final slide, save preferences
                        if (nextSlide === 6) {
                            this.savePreferences();
                        }
                        
                        this.goToSlide(nextSlide);
                    }
                });
            });
            
            // Add click event listeners to previous buttons
            this.prevButtons.forEach(button => {
                button.addEventListener('click', () => {
                    const prevSlide = parseInt(button.dataset.prev);
                    if (prevSlide < this.activeSlide) {
                        this.goToSlide(prevSlide);
                    }
                });
            });
            
            // Tutorial modal controls
            if (this.closeTutorial) {
                this.closeTutorial.addEventListener('click', () => {
                    this.hideTutorial();
                });
            }
            
            if (this.completeTutorial) {
                this.completeTutorial.addEventListener('click', () => {
                    this.hideTutorial();
                    // Mark tutorial as completed
                    fetch('/api/onboarding', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            tutorial_completed: true
                        })
                    });
                });
            }
        },
        
        // Update active slide
        updateActiveSlide: function(slideNumber) {
            // Update steps
            this.steps.forEach(step => {
                const stepNumber = parseInt(step.dataset.step);
                step.classList.remove('active', 'completed');
                
                if (stepNumber === slideNumber) {
                    step.classList.add('active');
                } else if (stepNumber < slideNumber) {
                    step.classList.add('completed');
                }
            });
            
            // Update slides
            if (this.slideWrapper) {
                this.slideWrapper.style.transform = `translateX(-${(slideNumber - 1) * 100}%)`;
            }
            
            // Update active slide number
            this.activeSlide = slideNumber;
        },
        
        // Go to specific slide
        goToSlide: function(slideNumber) {
            this.updateActiveSlide(slideNumber);
        },
        
        // Enable next button for a specific slide
        enableNextButton: function(slideNumber) {
            const nextButton = document.querySelector(`.slide[data-slide="${slideNumber}"] .nav-button-next`);
            if (nextButton) {
                nextButton.removeAttribute('disabled');
            }
        },
        
        // Show tutorial modal
        showTutorial: function() {
            if (this.tutorialModal) {
                this.tutorialModal.classList.add('show');
            }
        },
        
        // Hide tutorial modal
        hideTutorial: function() {
            if (this.tutorialModal) {
                this.tutorialModal.classList.remove('show');
            }
        },
        
        // Save user preferences
        savePreferences: function() {
            // Set default values for any unselected preferences
            if (!this.userPreferences.experienceLevel) this.userPreferences.experienceLevel = 'intermediate';
            if (!this.userPreferences.responseLength) this.userPreferences.responseLength = 'balanced';
            if (!this.userPreferences.modelType) this.userPreferences.modelType = 'enhanced';
            if (this.userPreferences.showTutorial === null) this.userPreferences.showTutorial = false;
            
            // Send preferences to the server
            fetch('/api/onboarding', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    preferences: this.userPreferences,
                    tutorial_completed: !this.userPreferences.showTutorial
                })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Preferences saved:', data);
            })
            .catch(error => {
                console.error('Error saving preferences:', error);
            });
        }
    };
    
    // Export helpers to global scope
    window.GAKR = window.GAKR || {};
    window.GAKR.tutorialHelper = tutorialHelper;
    window.GAKR.chatHelper = chatHelper;
    window.GAKR.onboardingHelper = onboardingHelper;
    
    // Initialize chat on chat page
    if (document.querySelector('.gemini-chat-layout')) {
        chatHelper.init('chatContainer', 'userInput', 'submitButton');
    }
    
    // Initialize onboarding on onboarding page
    if (document.querySelector('.onboarding-layout')) {
        onboardingHelper.init();
    }
});