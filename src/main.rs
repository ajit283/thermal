use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, Role,
    },
};
use crossterm::{
    cursor, // Import cursor module for Show/Hide
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    },
    execute,
    terminal::{
        Clear as TerminalClear, ClearType, EnterAlternateScreen, LeaveAlternateScreen,
        disable_raw_mode, enable_raw_mode,
    },
};
use futures::StreamExt;
use ratatui::{prelude::*, widgets::*};
use serde::Deserialize;
use std::{
    collections::HashMap,
    env,
    error::Error,
    fs,
    io::{self, Stdout, Write},
    path::{Path, PathBuf},
    process::Command as StdCommand,
    time::{Duration, Instant as StdInstant},
};
use tempfile::Builder;
use tempfile::NamedTempFile;
use tokio::sync::mpsc;

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, Result as RusqliteResult, params};
use uuid::Uuid;

const APP_TITLE: &str = "Ratatui OpenAI Chat";
const HIGHLIGHT_SYMBOL: &str = "> ";
const DB_FILE_NAME: &str = "chat_history.db";
const MAX_TITLE_LEN: usize = 50;

// --- Structs for TOML Configuration ---
#[derive(Deserialize, Debug, Default, Clone)]
struct EndpointsTomlConfig {
    default_openai_endpoint: Option<String>,
    openai_endpoints: Option<HashMap<String, OpenAIEndpointToml>>,
}

#[derive(Deserialize, Debug, Clone)]
struct OpenAIEndpointToml {
    api_key: Option<String>,
    api_base: Option<String>,
    default_model: Option<String>,
}

// --- load_toml_config function ---
fn load_toml_config(path: &Path) -> Result<EndpointsTomlConfig, Box<dyn Error>> {
    if !path.exists() {
        return Ok(EndpointsTomlConfig::default());
    }
    let content = std::fs::read_to_string(path)?;
    let config: EndpointsTomlConfig = toml::from_str(&content)?;
    Ok(config)
}

// --- Application State and Messages ---
#[derive(Clone)]
struct AppMessage {
    role: Role,
    content: String,
    timestamp: DateTime<Utc>,
}

enum AppUpdate {
    AssistantChunk(String),
    AssistantError(String),
    AssistantDone,
}

enum AppMode {
    Chatting,
    PickingConversation,
}

enum AppAction {
    None,
    LaunchEditor(String),
}

#[derive(Clone, Debug)]
struct ConversationMeta {
    id: String,
    title: String,
    updated_at: DateTime<Utc>,
}

struct App<'a> {
    input: String,
    messages: Vec<AppMessage>,
    openai_client: Client<OpenAIConfig>,
    model_to_use: String,
    is_loading: bool,
    config_source_message: String,
    update_sender: mpsc::Sender<AppUpdate>,
    update_receiver: mpsc::Receiver<AppUpdate>,
    message_list_state: ListState,
    input_cursor_position: usize,
    status_message: Option<String>,
    theme: Theme<'a>,
    db_conn: Connection,
    app_mode: AppMode,
    current_conversation_id: Option<String>,
    picker_items: Vec<ConversationMeta>,
    picker_state: ListState,
}

struct Theme<'a> {
    base_style: Style,
    user_style: Style,
    assistant_style: Style,
    input_block_title_style: Style,
    chat_block_title_style: Style,
    loading_style: Style,
    status_style: Style,
    highlight_style: Style,
    picker_title_style: Style,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl Default for Theme<'_> {
    fn default() -> Self {
        Theme {
            base_style: Style::default().fg(Color::White).bg(Color::Black),
            user_style: Style::default().fg(Color::Cyan),
            assistant_style: Style::default().fg(Color::Green),
            input_block_title_style: Style::default().fg(Color::Yellow),
            chat_block_title_style: Style::default().fg(Color::Yellow),
            loading_style: Style::default().fg(Color::Magenta),
            status_style: Style::default().fg(Color::Gray),
            highlight_style: Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
            picker_title_style: Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
            _phantom: std::marker::PhantomData,
        }
    }
}

// --- Database Functions ---
fn get_db_path() -> Result<PathBuf, Box<dyn Error>> {
    let mut path = dirs::data_dir()
        .ok_or_else(|| Box::<dyn Error>::from("Could not find user data directory"))?
        .join("ratatui_openai_chat");
    fs::create_dir_all(&path)?;
    path.push(DB_FILE_NAME);
    Ok(path)
}

fn initialize_database(conn: &Connection) -> RusqliteResult<()> {
    conn.execute_batch(
        "BEGIN;
        CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, timestamp TEXT NOT NULL, message_order INTEGER NOT NULL, FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages (conversation_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations (updated_at);
        COMMIT;"
    )?;
    Ok(())
}

// --- Editor Function (blocking) ---
fn open_content_in_editor_blocking(content: &str) -> String {
    let editor_env = env::var("EDITOR");
    let vis_env = env::var("VISUAL");

    let editor = vis_env.or(editor_env).unwrap_or_else(|_| {
        if cfg!(target_os = "windows") {
            "notepad".to_string()
        } else {
            "vi".to_string()
        }
    });

    match Builder::new()
        .prefix("chat_edit_") // Set the prefix
        .suffix(".md") // Set the suffix
        .tempfile()
    {
        Ok(mut temp_file) => {
            if let Err(e) = temp_file.write_all(content.as_bytes()) {
                return format!("Failed to write to temp file: {}", e);
            }

            let temp_file_path = temp_file.path().to_path_buf();
            let temp_file_path_str = temp_file_path.to_string_lossy();

            let mut command;
            if cfg!(target_os = "windows") {
                command = StdCommand::new("cmd");
                command.args(["/C", "start", "/WAIT", "\"\"", &editor, &temp_file_path_str]);
            } else {
                command = StdCommand::new("sh");
                let shell_command = format!("{} '{}'", editor, temp_file_path_str);
                command.arg("-c").arg(shell_command);
            }

            let status_msg = match command.status() {
                Ok(status) => {
                    if status.success() {
                        format!("Editor closed (file: {})", temp_file_path.display())
                    } else {
                        format!("Editor '{}' exited with status: {}", editor, status)
                    }
                }
                Err(e) => {
                    format!("Failed to launch editor '{}': {}", editor, e)
                }
            };
            status_msg
        }
        Err(e) => {
            format!("Failed to create temp file: {}", e)
        }
    }
}

impl<'a> App<'a> {
    fn new(
        client: Client<OpenAIConfig>,
        model: String,
        config_source: String,
        db_conn: Connection,
    ) -> Self {
        let (tx, rx) = mpsc::channel(100);
        App {
            input: String::new(),
            messages: Vec::new(),
            openai_client: client,
            model_to_use: model,
            is_loading: false,
            config_source_message: config_source,
            update_sender: tx,
            update_receiver: rx,
            message_list_state: ListState::default(),
            input_cursor_position: 0,
            status_message: None,
            theme: Theme::default(),
            db_conn,
            app_mode: AppMode::Chatting,
            current_conversation_id: None,
            picker_items: Vec::new(),
            picker_state: ListState::default(),
        }
    }

    fn create_new_conversation(
        &mut self,
        first_message_content: &str,
    ) -> Result<(), Box<dyn Error>> {
        let conv_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let title = first_message_content
            .chars()
            .take(MAX_TITLE_LEN)
            .collect::<String>();
        self.db_conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)",
            params![conv_id, title, now.to_rfc3339(), now.to_rfc3339()],
        )?;
        self.current_conversation_id = Some(conv_id);
        Ok(())
    }

    fn add_message_to_db(
        &self,
        conv_id: &str,
        role: Role,
        content: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<(), Box<dyn Error>> {
        let role_str = match role {
            Role::User => "user",
            Role::Assistant => "assistant",
            _ => "system",
        };
        let message_order: i64 = self.db_conn.query_row(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?1",
            params![conv_id],
            |row| row.get(0),
        )?;
        self.db_conn.execute("INSERT INTO messages (conversation_id, role, content, timestamp, message_order) VALUES (?1, ?2, ?3, ?4, ?5)", params![conv_id, role_str, content, timestamp.to_rfc3339(), message_order])?;
        self.db_conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![Utc::now().to_rfc3339(), conv_id],
        )?;
        Ok(())
    }

    fn submit_message(&mut self) {
        let trimmed_input = self.input.trim();
        if trimmed_input.is_empty() || self.is_loading {
            return;
        }
        let user_message_content = trimmed_input.to_string();
        let now = Utc::now();
        if self.current_conversation_id.is_none() {
            if let Err(e) = self.create_new_conversation(&user_message_content) {
                self.status_message = Some(format!("Error creating conversation: {}", e));
                return;
            }
        }
        let current_conv_id = self.current_conversation_id.as_ref().unwrap().clone();
        self.messages.push(AppMessage {
            role: Role::User,
            content: user_message_content.clone(),
            timestamp: now,
        });
        if let Err(e) =
            self.add_message_to_db(&current_conv_id, Role::User, &user_message_content, now)
        {
            self.status_message = Some(format!("Error saving message: {}", e));
        }
        self.message_list_state
            .select(Some(self.messages.len().saturating_sub(1)));
        self.input.clear();
        self.input_cursor_position = 0;
        self.is_loading = true;
        self.status_message = Some("Sending to AI...".to_string());
        let history_for_api: Vec<ChatCompletionRequestMessage> = self
            .messages
            .iter()
            .filter_map(|msg| match msg.role {
                Role::User => Some(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(msg.content.clone())
                        .build()
                        .unwrap()
                        .into(),
                ),
                Role::Assistant => Some(
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(msg.content.clone())
                        .build()
                        .unwrap()
                        .into(),
                ),
                _ => None,
            })
            .collect();
        let client = self.openai_client.clone();
        let model = self.model_to_use.clone();
        let sender = self.update_sender.clone();
        tokio::spawn(async move {
            let request = CreateChatCompletionRequestArgs::default()
                .model(&model)
                .messages(history_for_api)
                .build();
            if let Err(e) = request {
                let _ = sender
                    .send(AppUpdate::AssistantError(format!(
                        "Request build error: {}",
                        e
                    )))
                    .await;
                let _ = sender.send(AppUpdate::AssistantDone).await;
                return;
            }
            let stream_result = client.chat().create_stream(request.unwrap()).await;
            match stream_result {
                Ok(mut stream) => {
                    while let Some(chunk) = stream.next().await {
                        match chunk {
                            Ok(response) => {
                                for choice in response.choices {
                                    if let Some(content) = choice.delta.content {
                                        if sender
                                            .send(AppUpdate::AssistantChunk(content))
                                            .await
                                            .is_err()
                                        {
                                            return;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = sender.send(AppUpdate::AssistantError(e.to_string())).await;
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = sender
                        .send(AppUpdate::AssistantError(format!(
                            "Stream creation error: {}",
                            e
                        )))
                        .await;
                }
            }
            let _ = sender.send(AppUpdate::AssistantDone).await;
        });
    }

    fn handle_chatting_input(
        &mut self,
        key_code: KeyCode,
        key_modifiers: KeyModifiers,
    ) -> AppAction {
        if key_modifiers == KeyModifiers::CONTROL && key_code == KeyCode::Char('e') {
            let content_to_edit = if let Some(selected_idx) = self.message_list_state.selected() {
                self.messages
                    .get(selected_idx)
                    .map(|msg| msg.content.clone())
            } else if self.is_loading {
                self.messages
                    .last()
                    .filter(|msg| msg.role == Role::Assistant)
                    .map(|msg| msg.content.clone())
            } else {
                None
            };

            if let Some(content) = content_to_edit {
                self.status_message = Some("Preparing to open editor...".to_string());
                return AppAction::LaunchEditor(content);
            } else {
                self.status_message =
                    Some("No message selected or AI streaming to edit.".to_string());
                return AppAction::None;
            }
        }

        match key_code {
            KeyCode::Enter => self.submit_message(),
            KeyCode::Char(c) => {
                self.input.insert(self.input_cursor_position, c);
                self.input_cursor_position += 1;
            }
            KeyCode::Backspace => {
                if self.input_cursor_position > 0 {
                    self.input_cursor_position -= 1;
                    self.input.remove(self.input_cursor_position);
                }
            }
            KeyCode::Left => {
                if self.input_cursor_position > 0 {
                    self.input_cursor_position -= 1;
                }
            }
            KeyCode::Right => {
                if self.input_cursor_position < self.input.len() {
                    self.input_cursor_position += 1;
                }
            }
            KeyCode::Up => {
                if !self.messages.is_empty() {
                    match self.message_list_state.selected() {
                        Some(idx) if idx > 0 => self.message_list_state.select(Some(idx - 1)),
                        Some(_) => self.message_list_state.select(Some(0)),
                        None => self
                            .message_list_state
                            .select(Some(self.messages.len() - 1)),
                    }
                }
            }
            KeyCode::Down => {
                if !self.messages.is_empty() {
                    match self.message_list_state.selected() {
                        Some(idx) if idx < self.messages.len() - 1 => {
                            self.message_list_state.select(Some(idx + 1))
                        }
                        Some(idx) => self.message_list_state.select(Some(idx)),
                        None => self.message_list_state.select(Some(0)),
                    }
                }
            }
            _ => {}
        }
        AppAction::None
    }

    fn load_conversations_for_picker(&mut self) -> Result<(), Box<dyn Error>> {
        let mut stmt = self.db_conn.prepare(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT 50",
        )?;
        let iter = stmt.query_map([], |row| {
            Ok(ConversationMeta {
                id: row.get(0)?,
                title: row.get(1)?,
                updated_at: row
                    .get::<_, String>(2)?
                    .parse::<DateTime<Utc>>()
                    .unwrap_or_else(|_| Utc::now()),
            })
        })?;
        self.picker_items.clear();
        self.picker_items.push(ConversationMeta {
            id: "NEW_CHAT".to_string(),
            title: "[+] New Chat".to_string(),
            updated_at: Utc::now(),
        });
        for item_result in iter {
            if let Ok(item) = item_result {
                self.picker_items.push(item);
            } else if let Err(e) = item_result {
                self.status_message = Some(format!("Err load convo: {}", e));
            }
        }
        self.picker_state.select(Some(0));
        Ok(())
    }

    fn load_selected_conversation(&mut self, conv_id: &str) -> Result<(), Box<dyn Error>> {
        self.messages.clear();
        let mut stmt = self.db_conn.prepare("SELECT role, content, timestamp FROM messages WHERE conversation_id = ?1 ORDER BY message_order ASC")?;
        let iter = stmt.query_map(params![conv_id], |row| {
            let role_str: String = row.get(0)?;
            let role = match role_str.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                _ => Role::System,
            };
            Ok(AppMessage {
                role,
                content: row.get(1)?,
                timestamp: row
                    .get::<_, String>(2)?
                    .parse::<DateTime<Utc>>()
                    .unwrap_or_else(|_| Utc::now()),
            })
        })?;
        for msg_result in iter {
            if let Ok(msg) = msg_result {
                self.messages.push(msg);
            } else if let Err(e) = msg_result {
                self.status_message = Some(format!("Err load msg: {}", e));
            }
        }
        self.current_conversation_id = Some(conv_id.to_string());
        self.message_list_state.select(if self.messages.is_empty() {
            None
        } else {
            Some(self.messages.len() - 1)
        });
        Ok(())
    }

    fn handle_picker_input(&mut self, key_code: KeyCode) {
        match key_code {
            KeyCode::Esc => {
                self.app_mode = AppMode::Chatting;
            }
            KeyCode::Up => {
                if !self.picker_items.is_empty() {
                    let current = self.picker_state.selected().unwrap_or(0);
                    self.picker_state.select(Some(current.saturating_sub(1)));
                }
            }
            KeyCode::Down => {
                if !self.picker_items.is_empty() {
                    let current = self.picker_state.selected().unwrap_or(0);
                    if current < self.picker_items.len() - 1 {
                        self.picker_state.select(Some(current + 1));
                    }
                }
            }
            KeyCode::Enter => {
                if let Some(idx) = self.picker_state.selected() {
                    if idx < self.picker_items.len() {
                        let item = self.picker_items[idx].clone();
                        if item.id == "NEW_CHAT" {
                            self.messages.clear();
                            self.current_conversation_id = None;
                            self.input.clear();
                            self.message_list_state.select(None);
                            self.status_message = Some("New chat started.".to_string());
                        } else {
                            if let Err(e) = self.load_selected_conversation(&item.id) {
                                self.status_message = Some(format!("Err load chat: {}", e));
                            } else {
                                self.status_message = Some(format!("Loaded: {}", item.title));
                            }
                        }
                        self.app_mode = AppMode::Chatting;
                    }
                }
            }
            _ => {}
        }
    }

    fn update_from_channel(&mut self) {
        match self.update_receiver.try_recv() {
            Ok(AppUpdate::AssistantChunk(chunk)) => {
                let mut is_new_msg = false;
                if self
                    .messages
                    .last()
                    .map_or(true, |m| m.role != Role::Assistant)
                    || self.messages.is_empty()
                {
                    if !chunk.is_empty() {
                        self.messages.push(AppMessage {
                            role: Role::Assistant,
                            content: chunk,
                            timestamp: Utc::now(),
                        });
                        is_new_msg = true;
                    }
                } else if let Some(last) = self.messages.last_mut() {
                    if last.role == Role::Assistant {
                        last.content.push_str(&chunk);
                    } else if !chunk.is_empty() {
                        self.messages.push(AppMessage {
                            role: Role::Assistant,
                            content: chunk,
                            timestamp: Utc::now(),
                        });
                        is_new_msg = true;
                    }
                }
                self.status_message = Some("AI Typing...".to_string());
                if is_new_msg && !self.messages.is_empty() {
                    self.message_list_state
                        .select(Some(self.messages.len() - 1));
                }
            }
            Ok(AppUpdate::AssistantError(err_msg)) => {
                let now = Utc::now();
                let content = format!("[Error]: {}", err_msg);
                self.messages.push(AppMessage {
                    role: Role::Assistant,
                    content: content.clone(),
                    timestamp: now,
                });
                if let Some(id) = &self.current_conversation_id {
                    if let Err(e) = self.add_message_to_db(id, Role::Assistant, &content, now) {
                        self.status_message = Some(format!("Err save err: {}", e));
                    }
                }
                if !self.messages.is_empty() {
                    self.message_list_state
                        .select(Some(self.messages.len() - 1));
                }
                self.is_loading = false;
                self.status_message = Some(format!(
                    "Error: {}",
                    err_msg.chars().take(50).collect::<String>()
                ));
            }
            Ok(AppUpdate::AssistantDone) => {
                self.is_loading = false;
                self.status_message = Some("Ready.".to_string());
                if let Some(id) = &self.current_conversation_id {
                    if let Some(last) = self.messages.last() {
                        if last.role == Role::Assistant && !last.content.is_empty() {
                            if let Err(e) = self.add_message_to_db(
                                id,
                                Role::Assistant,
                                &last.content,
                                last.timestamp,
                            ) {
                                self.status_message = Some(format!("Err save AI resp: {}", e));
                            }
                        }
                    }
                }
                if self
                    .messages
                    .last()
                    .map_or(false, |m| m.role == Role::Assistant && m.content.is_empty())
                {
                    self.messages.pop();
                    if self.messages.is_empty() {
                        self.message_list_state.select(None);
                    } else {
                        self.message_list_state
                            .select(Some(self.messages.len() - 1));
                    }
                } else if !self.messages.is_empty()
                    && self
                        .messages
                        .last()
                        .map_or(false, |m| m.role == Role::Assistant)
                {
                    if self.message_list_state.selected() != Some(self.messages.len() - 1) {
                        self.message_list_state
                            .select(Some(self.messages.len() - 1));
                    }
                }
            }
            Err(mpsc::error::TryRecvError::Empty) => {}
            Err(mpsc::error::TryRecvError::Disconnected) => {
                self.is_loading = false;
                self.status_message = Some("AI worker connection lost.".to_string());
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let mut chosen_endpoint_name_from_cli: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--endpoint" {
            if i + 1 < args.len() {
                chosen_endpoint_name_from_cli = Some(args[i + 1].clone());
                break;
            } else {
                eprintln!("Warning: --endpoint flag provided without a value.");
            }
        }
        i += 1;
    }
    let toml_path = Path::new("endpoints.toml");
    let toml_config = load_toml_config(toml_path).unwrap_or_else(|err| {
        eprintln!(
            "Warning: Could not load '{}': {}. Using defaults.",
            toml_path.display(),
            err
        );
        EndpointsTomlConfig::default()
    });
    let mut selected_endpoint_details: Option<OpenAIEndpointToml> = None;
    let mut config_source_message: String = "Defaults (e.g. OPENAI_API_KEY env var)".to_string();
    let endpoint_name_to_load: Option<String> = chosen_endpoint_name_from_cli
        .clone()
        .or_else(|| toml_config.default_openai_endpoint.clone());
    if let Some(name) = &endpoint_name_to_load {
        if let Some(endpoints_map) = &toml_config.openai_endpoints {
            if let Some(details) = endpoints_map.get(name) {
                selected_endpoint_details = Some(details.clone());
                config_source_message = format!("endpoint '{}' from TOML", name);
            } else {
                eprintln!("Warning: Endpoint '{}' not found in TOML.", name);
            }
        } else if chosen_endpoint_name_from_cli.is_some()
            || toml_config.default_openai_endpoint.is_some()
        {
            eprintln!(
                "Warning: Endpoint '{}' specified, but no [openai_endpoints] table in TOML.",
                name
            );
        }
    } else if let Some(endpoints_map) = &toml_config.openai_endpoints {
        if let Some((name, details)) = endpoints_map.iter().next() {
            selected_endpoint_details = Some(details.clone());
            config_source_message = format!("first available endpoint '{}' from TOML", name);
        }
    }
    let mut openai_client_config = OpenAIConfig::new();
    let mut model_to_use = "gpt-4o".to_string();
    if let Some(details) = &selected_endpoint_details {
        if let Some(key) = &details.api_key {
            openai_client_config = openai_client_config.with_api_key(key);
        }
        if let Some(base) = &details.api_base {
            openai_client_config = openai_client_config.with_api_base(base);
        }
        if let Some(model) = &details.default_model {
            model_to_use = model.clone();
        }
    }
    let client = Client::with_config(openai_client_config);
    println!(
        "OpenAI client: {}. Model: {}.",
        &config_source_message, &model_to_use
    );
    let db_path = get_db_path()?;
    println!("DB at: {}", db_path.display());
    let db_conn = Connection::open(&db_path)
        .map_err(|e| format!("DB open error {}: {}", db_path.display(), e))?;
    initialize_database(&db_conn).map_err(|e| format!("DB init error: {}", e))?;
    println!("DB initialized.");
    println!("Starting TUI...");

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(client, model_to_use, config_source_message, db_conn);
    app.status_message =
        Some("Ctrl+O: Open chats. Ctrl+E: Edit message. Enter: Send. Ctrl+C: Quit.".to_string());

    let res = run_app(&mut terminal, &mut app).await;

    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?; // Use fresh io::stdout()
    terminal.show_cursor()?;
    if let Err(err) = res {
        eprintln!("App error: {:?}", err);
    }
    Ok(())
}

async fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App<'_>) -> io::Result<()> {
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = StdInstant::now();

    loop {
        app.update_from_channel();
        let mut pending_editor_action: Option<String> = None;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if key.modifiers == KeyModifiers::CONTROL {
                        match key.code {
                            KeyCode::Char('c') => return Ok(()),
                            KeyCode::Char('o') => {
                                app.app_mode = AppMode::PickingConversation;
                                if let Err(e) = app.load_conversations_for_picker() {
                                    app.status_message = Some(format!("Err load convos: {}", e));
                                    app.app_mode = AppMode::Chatting;
                                }
                                terminal.draw(|f| ui(f, app))?; // Draw picker immediately
                                last_tick = StdInstant::now(); // Reset tick after picker draw
                                continue;
                            }
                            _ => {}
                        }
                    }

                    let action_result = match app.app_mode {
                        AppMode::Chatting => app.handle_chatting_input(key.code, key.modifiers),
                        AppMode::PickingConversation => {
                            app.handle_picker_input(key.code);
                            AppAction::None
                        }
                    };

                    if let AppAction::LaunchEditor(content) = action_result {
                        pending_editor_action = Some(content);
                    }
                }
            }
        }

        if let Some(content_to_edit) = pending_editor_action.take() {
            let mut stdout_handle = io::stdout();
            execute!(
                stdout_handle,
                LeaveAlternateScreen,
                DisableMouseCapture,
                cursor::Show
            )?;
            stdout_handle.flush()?;
            disable_raw_mode()?;

            let editor_status_msg = open_content_in_editor_blocking(&content_to_edit);

            enable_raw_mode()?;
            let mut stdout_resume_handle = io::stdout();
            execute!(
                stdout_resume_handle,
                EnterAlternateScreen,
                EnableMouseCapture,
                cursor::Hide
            )?;
            stdout_resume_handle.flush()?;

            terminal.clear()?; // Use Ratatui's clear method
            app.status_message = Some(editor_status_msg);

            terminal.draw(|f| ui(f, app))?;
            last_tick = StdInstant::now();
        } else {
            // Normal draw if no editor action this iteration OR if tick has elapsed
            if last_tick.elapsed() >= tick_rate || timeout == Duration::from_secs(0) {
                // Also draw if poll timed out immediately
                terminal.draw(|f| ui(f, app))?;
                last_tick = StdInstant::now();
            }
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    f.render_widget(Clear, f.size());
    match app.app_mode {
        AppMode::Chatting => ui_chatting(f, app),
        AppMode::PickingConversation => ui_picker(f, app),
    }
}

fn ui_chatting(f: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Min(0),
                Constraint::Length(1),
                Constraint::Length(3),
            ]
            .as_ref(),
        )
        .split(f.size());
    let chat_area_width = chunks[0].width;
    let chat_pane_height = chunks[0].height;
    let highlight_symbol_len = HIGHLIGHT_SYMBOL.len() as u16;
    let max_message_block_height = (chat_pane_height / 3).max(1);
    let display_messages: Vec<ListItem> = app
        .messages
        .iter()
        .map(|msg| {
            let style = if msg.role == Role::User {
                theme.user_style
            } else {
                theme.assistant_style
            };
            let prefix_str = match msg.role {
                Role::User => "You: ",
                Role::Assistant => "AI:  ",
                _ => "Sys: ",
            };
            let prefix_len = prefix_str.len() as u16;
            let indentation = " ".repeat(prefix_str.len());
            let content_wrap_width = chat_area_width
                .saturating_sub(prefix_len)
                .saturating_sub(highlight_symbol_len);
            let wrapped_strings = textwrap::wrap(&msg.content, content_wrap_width.max(1) as usize);
            let mut all_lines: Vec<Line> = vec![];
            let prefix_span = Span::styled(prefix_str, style.add_modifier(Modifier::BOLD));
            if let Some(first) = wrapped_strings.first() {
                all_lines.push(Line::from(vec![
                    prefix_span.clone(),
                    Span::styled(first.clone(), style),
                ]));
            } else if !msg.content.is_empty() {
                all_lines.push(Line::from(vec![
                    prefix_span.clone(),
                    Span::styled(msg.content.clone(), style),
                ]));
            } else {
                all_lines.push(Line::from(prefix_span.clone()));
            }
            for line in wrapped_strings.iter().skip(1) {
                all_lines.push(Line::from(vec![
                    Span::raw(indentation.clone()),
                    Span::styled(line.clone(), style),
                ]));
            }
            if all_lines.is_empty() {
                all_lines.push(Line::from(Span::styled(
                    prefix_str,
                    style.add_modifier(Modifier::BOLD),
                )));
            }
            let final_lines =
                if all_lines.len() as u16 > max_message_block_height && chat_pane_height > 0 {
                    all_lines
                        .clone()
                        .into_iter()
                        .skip(
                            all_lines
                                .len()
                                .saturating_sub(max_message_block_height as usize),
                        )
                        .collect()
                } else {
                    all_lines
                };
            ListItem::new(Text::from(
                if final_lines.is_empty() && !wrapped_strings.is_empty() && chat_pane_height > 0 {
                    vec![Line::from("...")]
                } else {
                    final_lines
                },
            ))
        })
        .collect();
    let messages_list = List::new(display_messages)
        .block(
            Block::default()
                .title(Span::styled(
                    APP_TITLE,
                    theme.chat_block_title_style.clone(),
                ))
                .title_alignment(Alignment::Center),
        )
        .highlight_style(theme.highlight_style.clone())
        .highlight_symbol(HIGHLIGHT_SYMBOL);
    f.render_stateful_widget(messages_list, chunks[0], &mut app.message_list_state);
    let status_text = app
        .status_message
        .as_deref()
        .unwrap_or("Ctrl+O: Open. Ctrl+E: Edit. Enter: Send. Ctrl+C: Quit.");
    let status_bar = Paragraph::new(status_text)
        .style(if app.is_loading {
            theme.loading_style
        } else {
            theme.status_style
        })
        .alignment(Alignment::Left);
    f.render_widget(status_bar, chunks[1]);
    let conv_title = app
        .current_conversation_id
        .as_ref()
        .and_then(|id| {
            app.db_conn
                .query_row(
                    "SELECT title FROM conversations WHERE id = ?1",
                    params![id],
                    |row| row.get(0),
                )
                .optional()
                .ok()
                .flatten()
        })
        .unwrap_or_else(|| "New Chat".to_string());
    let input_title = format!(
        "To {} (ID: {})",
        conv_title.chars().take(20).collect::<String>(),
        app.current_conversation_id.as_deref().unwrap_or("N/A")
    );
    let input_paragraph = Paragraph::new(app.input.as_str())
        .style(if app.is_loading {
            theme.status_style
        } else {
            theme.base_style
        })
        .block(Block::default().borders(Borders::ALL).title(Span::styled(
            input_title,
            theme.input_block_title_style.clone(),
        )));
    f.render_widget(input_paragraph, chunks[2]);
    if !app.is_loading {
        f.set_cursor(
            chunks[2].x + app.input_cursor_position as u16 + 1,
            chunks[2].y + 1,
        );
    }
}

fn ui_picker(f: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let popup_area = centered_rect(70, 80, f.size());
    f.render_widget(Clear, popup_area);
    let items: Vec<ListItem> = app
        .picker_items
        .iter()
        .map(|item| {
            let title = Span::styled(item.title.clone(), theme.base_style);
            if item.id == "NEW_CHAT" {
                ListItem::new(Line::from(vec![title]))
            } else {
                let date = Span::styled(
                    item.updated_at.format(" (%y-%m-%d %H:%M)").to_string(),
                    theme.status_style.fg(Color::DarkGray),
                );
                ListItem::new(Line::from(vec![title, date]))
            }
        })
        .collect();
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " Open/New (Ctrl+O or Esc to close) ",
                    theme.picker_title_style.clone(),
                ))
                .title_alignment(Alignment::Center),
        )
        .highlight_style(theme.highlight_style.clone())
        .highlight_symbol(HIGHLIGHT_SYMBOL);
    f.render_stateful_widget(list, popup_area, &mut app.picker_state);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let ly = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Percentage((100 - percent_y) / 2),
                Constraint::Percentage(percent_y),
                Constraint::Percentage((100 - percent_y) / 2),
            ]
            .as_ref(),
        )
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints(
            [
                Constraint::Percentage((100 - percent_x) / 2),
                Constraint::Percentage(percent_x),
                Constraint::Percentage((100 - percent_x) / 2),
            ]
            .as_ref(),
        )
        .split(ly[1])[1]
}
