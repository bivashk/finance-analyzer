// AI Personal Finance Analyzer - Main Application Logic
// Author: Bivash Koirala
// Version: 1.0

class FinanceAnalyzer {
    constructor() {
        // Application state
        this.state = {
            conversations: [],
            currentConversation: [],
            transactions: [],
            memory: {
                summary: '',
                profile: '',
                keywordIndex: new Map()
            },
            settings: {
                apiKey: 'tgp_v1_o319HuxtA3Phy_VoWmflw1_WXLLKXZENiz2PbD8Q6UA',
                model: 'openai/gpt-oss-20b',
                temperature: 0.25,
                maxTokens: 800,
                keepMessages: 12,
                persistConversations: true,
                persistTransactions: true,
                persistMemory: true,
                theme: 'dark'
            }
        };

        // API configuration
        this.apiConfig = {
            baseUrl: 'https://api.together.xyz/v1/chat/completions',
            timeout: 20000,
            retryAttempts: 2,
            retryDelay: 1000
        };

        // Categories for transaction classification
        this.categories = [
            'Transport', 'Food & Drink', 'Shopping', 'Entertainment', 
            'Groceries', 'Utilities', 'Income', 'Travel', 'Electronics', 
            'Fuel', 'Health', 'Fitness', 'Rent', 'Education', 'Fees', 'Other'
        ];

        // System prompts
        this.prompts = {
            summary: "Summarize the conversation so far into a compact 'memory' that preserves user goals, constraints, key facts, preferences, and unresolved tasks. Keep it actionable and under 2000 characters.",
            categorization: `You are an expert transaction classifier. Allowed categories ONLY: ${this.categories.join(', ')}.
Input: N transaction descriptions, one per line.
Output: EXACTLY N lines, one category per line. No extra text.`,
            insights: `Given the following aggregated snapshot (INR-enabled formatting), produce:
1) 5–8 quick wins (bullets),
2) A 30-day action plan,
3) Risk flags, and
4) A budget split (percentages) with rationale.
Keep responses concise and high-impact.`
        };

        this.init();
    }

    // Initialize the application
    async init() {
        this.loadState();
        this.setupEventListeners();
        this.setupTabs();
        this.renderTransactions();
        this.updateDashboard();
        this.updateMemoryDisplay();
        this.populateCategoryFilter();
        
        // Load theme
        document.documentElement.classList.toggle('light', this.state.settings.theme === 'light');
        
        this.showToast('Welcome to FinanceAI! 🚀', 'success');
    }

    // Event Listeners Setup
    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Chat functionality
        document.getElementById('send-button').addEventListener('click', () => this.sendMessage());
        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        document.getElementById('clear-chat').addEventListener('click', () => this.clearChat());

        // Settings inputs
        document.getElementById('api-key').addEventListener('change', (e) => {
            this.state.settings.apiKey = e.target.value;
            this.saveState();
        });
        document.getElementById('model-select').addEventListener('change', (e) => {
            this.state.settings.model = e.target.value;
            this.saveState();
        });
        document.getElementById('temperature').addEventListener('change', (e) => {
            this.state.settings.temperature = parseFloat(e.target.value);
            this.saveState();
        });
        document.getElementById('max-tokens').addEventListener('change', (e) => {
            this.state.settings.maxTokens = parseInt(e.target.value);
            this.saveState();
        });

        // File upload
        document.getElementById('upload-button').addEventListener('click', () => {
            document.getElementById('csv-upload').click();
        });
        document.getElementById('csv-upload').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });

        // Quick actions
        document.getElementById('download-sample').addEventListener('click', () => this.downloadSampleCSV());
        document.getElementById('generate-insights').addEventListener('click', () => this.generateInsights());
        document.getElementById('rebuild-summary').addEventListener('click', () => this.rebuildSummary());
        document.getElementById('categorize-button').addEventListener('click', () => this.categorizeTransactions());

        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', () => this.toggleTheme());

        // Memory management
        document.getElementById('edit-memory').addEventListener('click', () => this.openMemoryEditor());
        document.getElementById('export-memory').addEventListener('click', () => this.exportMemory());
        document.getElementById('save-profile').addEventListener('click', () => this.saveProfile());

        // Modal handlers
        document.getElementById('cancel-memory-edit').addEventListener('click', () => this.closeMemoryEditor());
        document.getElementById('save-memory-edit').addEventListener('click', () => this.saveMemoryEdit());

        // Settings
        document.getElementById('export-all').addEventListener('click', () => this.exportAllData());
        document.getElementById('import-all').addEventListener('click', () => {
            document.getElementById('import-file').click();
        });
        document.getElementById('import-file').addEventListener('change', (e) => {
            this.importAllData(e.target.files[0]);
        });
        document.getElementById('clear-all').addEventListener('click', () => this.clearAllData());

        // Filters
        document.getElementById('search-transactions').addEventListener('input', (e) => {
            this.filterTransactions();
        });
        document.getElementById('category-filter').addEventListener('change', (e) => {
            this.filterTransactions();
        });

        // Persistence settings
        ['persist-conversations', 'persist-transactions', 'persist-memory'].forEach(id => {
            document.getElementById(id).addEventListener('change', (e) => {
                const setting = id.replace('persist-', '').replace('-', '');
                this.state.settings[`persist${setting.charAt(0).toUpperCase()}${setting.slice(1)}`] = e.target.checked;
                this.saveState();
            });
        });

        document.getElementById('keep-messages').addEventListener('change', (e) => {
            this.state.settings.keepMessages = parseInt(e.target.value);
            this.saveState();
        });
    }

    // Tab Management
    setupTabs() {
        this.switchTab('chat');
    }

    switchTab(tabName) {
        // Remove active class from all tabs and buttons
        document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.tab-button').forEach(button => button.classList.remove('active'));

        // Add active class to current tab and button
        document.getElementById(`${tabName}-tab`).classList.add('active');
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update dashboard when switching to it
        if (tabName === 'dashboard') {
            setTimeout(() => this.updateDashboard(), 100);
        }
    }

    // Chat Functionality
    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;

        input.value = '';
        this.addMessage('user', message);

        try {
            const response = await this.callAPI(message);
            this.addMessage('assistant', response);
            this.updateKeywordIndex(message);
            this.updateKeywordIndex(response);
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
            this.addMessage('assistant', `I apologize, but I encountered an error: ${error.message}`);
        }

        this.saveState();
    }

    addMessage(role, content) {
        const message = { role, content, timestamp: Date.now() };
        this.state.currentConversation.push(message);

        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role} flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content relative max-w-3xl';
        
        if (role === 'assistant') {
            contentDiv.innerHTML = this.formatMessage(content);
            this.addCopyButtons(contentDiv);
        } else {
            contentDiv.textContent = content;
        }

        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Update stats
        this.updateChatStats();
    }

    formatMessage(content) {
        // Basic markdown parsing
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            .replace(/\n/g, '<br>');
    }

    addCopyButtons(container) {
        const codeBlocks = container.querySelectorAll('pre');
        codeBlocks.forEach(block => {
            const copyButton = document.createElement('button');
            copyButton.className = 'copy-button';
            copyButton.textContent = 'Copy';
            copyButton.addEventListener('click', () => {
                navigator.clipboard.writeText(block.textContent);
                copyButton.textContent = 'Copied!';
                setTimeout(() => copyButton.textContent = 'Copy', 2000);
            });
            block.style.position = 'relative';
            block.appendChild(copyButton);
        });
    }

    updateChatStats() {
        const stats = document.getElementById('chat-stats');
        const messages = this.state.currentConversation.length;
        const tokens = this.estimateTokens();
        stats.textContent = `Messages: ${messages} | Est. Tokens: ${tokens}`;
    }

    estimateTokens() {
        return Math.ceil(this.state.currentConversation
            .map(m => m.content.length)
            .reduce((a, b) => a + b, 0) / 4);
    }

    clearChat() {
        if (confirm('Are you sure you want to clear the conversation?')) {
            this.state.currentConversation = [];
            document.getElementById('chat-messages').innerHTML = '';
            this.updateChatStats();
            this.saveState();
        }
    }

    // API Communication
    async callAPI(message) {
        const prompt = this.buildPrompt(message);
        const requestBody = {
            model: this.state.settings.model,
            messages: prompt,
            temperature: this.state.settings.temperature,
            max_tokens: this.state.settings.maxTokens,
            stream: false // Simplified for now
        };

        const response = await this.makeRequest(this.apiConfig.baseUrl, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.state.settings.apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (response.choices && response.choices[0]) {
            return response.choices[0].message.content;
        } else {
            throw new Error('Invalid API response');
        }
    }

    async makeRequest(url, options) {
        let lastError;
        
        for (let attempt = 0; attempt <= this.apiConfig.retryAttempts; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.apiConfig.timeout);

                const response = await fetch(url, {
                    ...options,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                return await response.json();
            } catch (error) {
                lastError = error;
                if (attempt < this.apiConfig.retryAttempts) {
                    await this.sleep(this.apiConfig.retryDelay * Math.pow(2, attempt));
                }
            }
        }

        throw lastError;
    }

    buildPrompt(userMessage) {
        const messages = [];
        
        // System message
        messages.push({
            role: 'system',
            content: 'You are a helpful AI assistant specializing in personal finance analysis and advice.'
        });

        // User profile if available
        if (this.state.memory.profile) {
            messages.push({
                role: 'system',
                content: `User Profile: ${this.state.memory.profile}`
            });
        }

        // Memory summary if available
        if (this.state.memory.summary) {
            messages.push({
                role: 'system',
                content: `Conversation Summary: ${this.state.memory.summary}`
            });
        }

        // Recent messages
        const recentMessages = this.state.currentConversation
            .slice(-this.state.settings.keepMessages)
            .map(msg => ({ role: msg.role, content: msg.content }));
        
        messages.push(...recentMessages);

        // Current user message
        messages.push({ role: 'user', content: userMessage });

        // Add relevant context from keyword search
        const relevantContext = this.searchRelevantContext(userMessage);
        if (relevantContext.length > 0) {
            messages.push({
                role: 'system',
                content: `Relevant context: ${relevantContext.join(' ')}`
            });
        }

        return messages;
    }

    // Context Management
    updateKeywordIndex(text) {
        const words = text.toLowerCase().match(/\b\w+\b/g) || [];
        words.forEach(word => {
            if (word.length > 3) { // Only index meaningful words
                if (!this.state.memory.keywordIndex.has(word)) {
                    this.state.memory.keywordIndex.set(word, []);
                }
                this.state.memory.keywordIndex.get(word).push({
                    text: text.substring(0, 200),
                    timestamp: Date.now()
                });
            }
        });
    }

    searchRelevantContext(query) {
        const queryWords = query.toLowerCase().match(/\b\w+\b/g) || [];
        const relevantSnippets = [];
        
        queryWords.forEach(word => {
            if (this.state.memory.keywordIndex.has(word)) {
                const snippets = this.state.memory.keywordIndex.get(word);
                relevantSnippets.push(...snippets.slice(-2)); // Get 2 most recent
            }
        });

        return relevantSnippets
            .sort((a, b) => b.timestamp - a.timestamp)
            .slice(0, 3)
            .map(s => s.text);
    }

    async rebuildSummary() {
        if (this.state.currentConversation.length === 0) {
            this.showToast('No conversation to summarize', 'warning');
            return;
        }

        try {
            const conversationText = this.state.currentConversation
                .map(msg => `${msg.role}: ${msg.content}`)
                .join('\n\n');

            const summaryResponse = await this.callAPI(`${this.prompts.summary}\n\nConversation:\n${conversationText}`);
            
            this.state.memory.summary = summaryResponse;
            this.updateMemoryDisplay();
            this.saveState();
            this.showToast('Memory summary rebuilt successfully', 'success');
        } catch (error) {
            this.showToast(`Error rebuilding summary: ${error.message}`, 'error');
        }
    }


    // Dashboard, AI and analytics methods to be implemented
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.financeAnalyzer = new FinanceAnalyzer();
});
