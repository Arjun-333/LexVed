document.addEventListener('DOMContentLoaded', () => {
    // ---------------------------------
    // 1. Wheel Data & Initialization
    // ---------------------------------
    const models = [
        { id: 'gpt4', name: 'GPT-4o', activeLabel: '(Active)', desc: 'Multimodal, Fast, Knowledge', type: 'img', src: 'https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg' },
        { id: 'gemini', name: 'Gemini 1.5 Pro', activeLabel: '', desc: '', type: 'icon', src: 'auto_awesome', color: '#4285F4' },
        { id: 'claude', name: 'Claude 3.5 Sonnet', activeLabel: '', desc: '', type: 'icon', src: 'edit_note', color: '#D9534F' },
        { id: 'llama', name: 'Llama 3 70B', activeLabel: '', desc: '', type: 'icon', src: 'all_inclusive', color: '#1877F2' },
        { id: 'mistral', name: 'Mistral Large 2', activeLabel: '', desc: '', type: 'icon', src: 'bolt', color: '#E4A11B' }
    ];

    // The wheel elements
    const wheelEl = document.getElementById('model-wheel');
    const itemHeight = 80; // matches CSS
    const radius = 150; // Distance to push panels back in 3D space
    const degreesPerPanel = 360 / models.length;

    let activeModelIndex = 1; // Default to Gemini 1.5 Pro
    let currentRotation = 0; // The continuous rotation of the wheel in degrees

    // Render items into the wheel
    models.forEach((model, i) => {
        const div = document.createElement('div');
        div.className = 'wheel-item';
        
        let iconHtml = model.type === 'img' 
            ? `<img src="${model.src}" alt="${model.name}">` 
            : `<span class="material-icons" style="color: ${model.color}; font-size: 2rem;">${model.src}</span>`;

        let descHtml = model.desc ? `<span class="model-desc">${model.desc}</span>` : '';
        
        div.innerHTML = `
            ${iconHtml}
            <div class="model-details">
                <span class="model-name">${model.name} <span class="active-lbl" style="display:none;">(Active)</span></span>
                ${descHtml}
            </div>
        `;
        
        // Arrange items in a circle by rotating them around X, then pushing them out by R
        const rx = i * degreesPerPanel;
        div.style.transform = `rotateX(${-rx}deg) translateZ(${radius}px)`;
        wheelEl.appendChild(div);
    });

    // ---------------------------------
    // 2. Wheel Drag & Snap Mechanics
    // ---------------------------------
    const wheelContainer = document.querySelector('.wheel-container');
    let isDragging = false;
    let startY = 0;
    let lastY = 0;
    let velocity = 0;
    let animationFrame;

    function applyRotation(rot) {
        wheelEl.style.transform = `rotateX(${rot}deg)`;
        
        // Determine the "active" physical item and highlight it
        // The rotation goes backwards as we drag down.
        // E.g., item 0 is at 0 degrees, item 1 is at 72 degrees.
        // Therefore, if the wheel rotation is 72, item 1 is upright.
        let normalized = rot % 360;
        if (normalized < 0) normalized += 360;
        
        // Which index is closest?
        // Index 0 requires rot 0. Index 1 requires rot 72.
        let closestIndex = Math.round(normalized / degreesPerPanel) % models.length;
        
        document.querySelectorAll('.wheel-item').forEach((item, i) => {
            const activeLbl = item.querySelector('.active-lbl');
            if (i === closestIndex) {
                item.classList.add('active-item');
                activeModelIndex = closestIndex;
                if(activeLbl) activeLbl.style.display = 'inline';
            } else {
                item.classList.remove('active-item');
                if(activeLbl) activeLbl.style.display = 'none';
            }
        });
    }

    // Snap the wheel perfectly to the active item
    function snapToActive() {
        cancelAnimationFrame(animationFrame);
        
        // Calculate the nearest target rotation
        // E.g. if currentRotation is 65, and degreesPerPanel is 72, nearest is 72.
        const nearestIndex = Math.round(currentRotation / degreesPerPanel);
        const targetRotation = nearestIndex * degreesPerPanel;
        
        // Small transition via CSS is applied automatically because we set 
        // transition: transform 0.6s in the CSS. We just set the target.
        currentRotation = targetRotation;
        applyRotation(currentRotation);
    }

    // Initialize position
    // Since Gemini is index 1, rotate the wheel to degreesPerPanel * 1
    currentRotation = 1 * degreesPerPanel;
    applyRotation(currentRotation);

    // Mouse events
    wheelContainer.addEventListener('mousedown', (e) => {
        isDragging = true;
        startY = e.clientY;
        lastY = e.clientY;
        velocity = 0;
        wheelEl.style.transition = 'none'; // disable transition while dragging
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const delta = e.clientY - lastY;
        lastY = e.clientY;
        
        // Map deltaY to rotation degrees (negative because dragging down rotates wheel up)
        const rotDelta = -(delta * 0.5); 
        currentRotation += rotDelta;
        velocity = rotDelta;
        
        applyRotation(currentRotation);
    });

    window.addEventListener('mouseup', () => {
        if (!isDragging) return;
        isDragging = false;
        wheelEl.style.transition = 'transform 0.5s cubic-bezier(0.2, 0.8, 0.2, 1)';
        
        // Add momentum based on velocity
        currentRotation += velocity * 5; 
        snapToActive();
    });

    // Also support wheel scrolling over the panel
    wheelContainer.addEventListener('wheel', (e) => {
        e.preventDefault();
        wheelEl.style.transition = 'none';
        currentRotation += (e.deltaY * 0.2);
        applyRotation(currentRotation);
        
        // Debounce snapping during scroll
        clearTimeout(window.scrollSnapTimer);
        window.scrollSnapTimer = setTimeout(() => {
            wheelEl.style.transition = 'transform 0.5s cubic-bezier(0.2, 0.8, 0.2, 1)';
            snapToActive();
        }, 150);
    });

    // ---------------------------------
    // 3. Chat Mechanics
    // ---------------------------------
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistory = document.getElementById('chat-history');

    function addMessage(text, isUser, modelBadge = '', metadata = {}) {
        const div = document.createElement('div');
        div.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        if (isUser) {
            div.innerHTML = `<p>${text}</p>`;
        } else {
            // Process citations if present
            let citationHtml = '';
            if (metadata.source) {
                citationHtml = `<div class="citation-box">Source: ${metadata.source}, Page: ${metadata.page}</div>`;
            }

            div.innerHTML = `
                <div class="bot-icon">
                    <span class="material-icons">gavel</span>
                </div>
                <div class="bubble">
                    <p>${text.replace(/\n/g, '<br>')}</p>
                    ${modelBadge ? `<span class="model-badge">Answered by ${modelBadge}</span>` : ''}
                </div>
            `;
        }
        
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // Also update dynamic placeholder
    function updatePlaceholder() {
        if (!models[activeModelIndex]) return;
        chatInput.placeholder = `Ask anything to ${models[activeModelIndex].name}...`;
    }
    
    // Periodically update placeholder in case active model changed
    setInterval(updatePlaceholder, 500);

    async function handleSend() {
        const text = chatInput.value.trim();
        if (!text) return;

        const currentModel = models[activeModelIndex];
        const currentModelName = currentModel.name;

        // Display User message
        addMessage(text, true);
        chatInput.value = '';

        // Add loading state
        const loadingId = 'loading-' + Date.now();
        const loadingHtml = `
            <div id="${loadingId}" class="message bot-message">
                <div class="bot-icon"><span class="material-icons">hourglass_empty</span></div>
                <div class="bubble"><p><i>Reasoning with ${currentModelName}...</i></p></div>
            </div>
        `;
        chatHistory.insertAdjacentHTML('beforeend', loadingHtml);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    provider: currentModel.provider
                })
            });
            const data = await res.json();
            
            // Remove loading msg
            const loadingEl = document.getElementById(loadingId);
            if(loadingEl) loadingEl.remove();

            if (data.error) {
                addMessage(`❌ Error: ${data.error}`, false);
            } else {
                addMessage(data.response, false, data.provider);
            }
        } catch (e) {
            const loadingEl = document.getElementById(loadingId);
            if(loadingEl) loadingEl.remove();
            addMessage(`❌ Request failed: ${e.message}`, false);
        }
    }

    sendBtn.addEventListener('click', handleSend);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSend();
    });

    // Allow suggestion chips to automatically fill and send
    document.querySelectorAll('.suggestion-chips button').forEach(btn => {
        btn.addEventListener('click', () => {
            chatInput.value = btn.innerText;
            handleSend();
        });
    });

    // ---------------------------------
    // 4. Light/Dark Theme Toggle
    // ---------------------------------
    const themeToggleBtn = document.getElementById('theme-toggle');
    const themeIcon = themeToggleBtn.querySelector('.material-icons');
    
    // Check local storage for theme
    const currentTheme = localStorage.getItem('theme') || 'light';
    if (currentTheme === 'dark') {
        document.body.classList.add('dark-theme');
        themeIcon.innerText = 'light_mode';
    }

    themeToggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('dark-theme');
        const isDark = document.body.classList.contains('dark-theme');
        
        if (isDark) {
            themeIcon.innerText = 'light_mode';
            localStorage.setItem('theme', 'dark');
        } else {
            themeIcon.innerText = 'dark_mode';
            localStorage.setItem('theme', 'light');
        }
    });

});
