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
            
            // Adjust if tooltip is off-screen
            const tooltipNewRect = tooltip.getBoundingClientRect();
            
            if (tooltipNewRect.left < 10) {
                tooltip.style.left = '10px';
            } else if (tooltipNewRect.right > window.innerWidth - 10) {
                tooltip.style.left = (window.innerWidth - tooltipNewRect.width - 10) + 'px';
            }
            
            if (tooltipNewRect.top < 10) {
                tooltip.style.top = '10px';
            } else if (tooltipNewRect.bottom > window.innerHeight - 10) {
                tooltip.style.top = (window.innerHeight - tooltipNewRect.height - 10) + 'px';
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
        
        // Complete the tutorial
        complete: function() {
            this.stop();
            this.onComplete();
            
            // Track that user has completed the tutorial
            this.trackTutorialProgress('completed');
        },
        
        // Track tutorial progress (analytics)
        trackTutorialProgress: function(action, stepNumber) {
            // This would normally send analytics data to a server
            console.log(`Tutorial ${action}${stepNumber ? ' - Step ' + stepNumber : ''}`);
        }
    };
    
    // Initialize tooltips, popovers, etc.
    const initTooltips = function() {
        // Initialize tooltips if Bootstrap is available
        if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function(tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
        
        // Initialize popovers if Bootstrap is available
        if (typeof bootstrap !== 'undefined' && bootstrap.Popover) {
            const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
            popoverTriggerList.map(function(popoverTriggerEl) {
                return new bootstrap.Popover(popoverTriggerEl);
            });
        }
    };
    
    // Initialize tooltips
    initTooltips();
    
    // Expose utilities to global scope for use in other scripts
    window.gakrUtils = {
        tutorialHelper: tutorialHelper,
        initTooltips: initTooltips
    };
});