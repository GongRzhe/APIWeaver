[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/gongrzhe-apiweaver-badge.png)](https://mseep.ai/app/gongrzhe-apiweaver)

# APIWeaver

A FastMCP server that dynamically creates MCP (Model Context Protocol) servers from web API configurations. This allows you to easily integrate any REST API, GraphQL endpoint, or web service into an MCP-compatible tool that can be used by AI assistants like Claude.

## Features

- 🚀 **Dynamic API Registration**: Register any web API at runtime
- 🔐 **Multiple Authentication Methods**: Bearer tokens, API keys, Basic auth, OAuth2, and custom headers
- 🛠️ **All HTTP Methods**: Support for GET, POST, PUT, DELETE, PATCH, and more
- 📝 **Flexible Parameters**: Query params, path params, headers, and request bodies
- 🔄 **Automatic Tool Generation**: Each API endpoint becomes an MCP tool
- 🧪 **Built-in Testing**: Test API connections before using them
- 📊 **Response Handling**: Automatic JSON parsing with fallback to text
- 🌐 **Multiple Transport Types**: STDIO, SSE, and Streamable HTTP transport support

## Transport Types

APIWeaver supports three different transport types to accommodate various deployment scenarios:

### STDIO Transport (Default)
- **Usage**: `apiweaver run` or `apiweaver run --transport stdio`
- **Best for**: Local tools, command-line usage, and MCP clients that connect via standard input/output
- **Characteristics**: Direct process communication, lowest latency, suitable for desktop applications
- **Endpoint**: N/A (uses stdin/stdout)

### SSE Transport (Legacy)
- **Usage**: `apiweaver run --transport sse --host 127.0.0.1 --port 8000`
- **Best for**: Legacy MCP clients that only support Server-Sent Events
- **Characteristics**: HTTP-based, one-way streaming from server to client
- **Endpoint**: `http://host:port/mcp`
- **Note**: This transport is deprecated in favor of Streamable HTTP

### Streamable HTTP Transport (Recommended)
- **Usage**: `apiweaver run --transport streamable-http --host 127.0.0.1 --port 8000`
- **Best for**: Modern web deployments, cloud environments, and new MCP clients
- **Characteristics**: Full HTTP-based communication, bidirectional streaming, better error handling
- **Endpoint**: `http://host:port/mcp`
- **Recommended**: This is the preferred transport for new deployments

## Installation

```bash
# Clone or download this repository
cd ~/Desktop/APIWeaver

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Claude Desktop

```json
{
  "mcpServers": {
    "apiweaver": {
      "command": "uvx",
      "args": ["apiweaver", "run"]
    }
  }
}
```

### Starting the Server

There are several ways to run the APIWeaver server with different transport types:

**1. After installation (recommended):**

If you have installed the package (e.g., using `pip install .` from the project root after installing requirements):

```bash
# Default STDIO transport
apiweaver run

# Streamable HTTP transport (recommended for web deployments)
apiweaver run --transport streamable-http --host 127.0.0.1 --port 8000

# SSE transport (legacy compatibility)
apiweaver run --transport sse --host 127.0.0.1 --port 8000
```

**2. Directly from the repository (for development):**

```bash
# From the root of the repository
python -m apiweaver.cli run [OPTIONS]
```

**Transport Options:**
- `--transport`: Choose from `stdio` (default), `sse`, or `streamable-http`
- `--host`: Host address for HTTP transports (default: 127.0.0.1)
- `--port`: Port for HTTP transports (default: 8000)
- `--path`: URL path for MCP endpoint (default: /mcp)

Run `apiweaver run --help` for all available options.

### Using with AI Assistants (like Claude Desktop)

APIWeaver is designed to expose web APIs as tools for AI assistants that support the Model Context Protocol (MCP). Here's how to use it:

1. **Start the APIWeaver Server:**
   
   **For modern MCP clients (recommended):**
   ```bash
   apiweaver run --transport streamable-http --host 127.0.0.1 --port 8000
   ```
   
   **For legacy compatibility:**
   ```bash
   apiweaver run --transport sse --host 127.0.0.1 --port 8000
   ```
   
   **For local desktop applications:**
   ```bash
   apiweaver run  # Uses STDIO transport
   ```

2. **Configure Your AI Assistant:**
   The MCP endpoint will be available at:
   - **Streamable HTTP**: `http://127.0.0.1:8000/mcp`
   - **SSE**: `http://127.0.0.1:8000/mcp`
   - **STDIO**: Direct process communication

3. **Register APIs and Use Tools:**
   Once connected, use the built-in `register_api` tool to define web APIs, then use the generated endpoint tools.

### Core Tools

The server provides these built-in tools:

1. **register_api** - Register a new API and create tools for its endpoints
2. **list_apis** - List all registered APIs and their endpoints
3. **unregister_api** - Remove an API and its tools
4. **test_api_connection** - Test connectivity to a registered API
5. **call_api** - Generic tool to call any registered API endpoint
6. **get_api_schema** - Get schema information for APIs and endpoints

### API Configuration Format

```json
{
  "name": "my_api",
  "base_url": "https://api.example.com",
  "description": "Example API integration",
  "auth": {
    "type": "bearer",
    "bearer_token": "your-token-here"
  },
  "headers": {
    "Accept": "application/json"
  },
  "endpoints": [
    {
      "name": "list_users",
      "description": "Get all users",
      "method": "GET",
      "path": "/users",
      "params": [
        {
          "name": "limit",
          "type": "integer",
          "location": "query",
          "required": false,
          "default": 10,
          "description": "Number of users to return"
        }
      ]
    }
  ]
}
```

## Examples

### Example 1: OpenWeatherMap API

```json
{
  "name": "weather",
  "base_url": "https://api.openweathermap.org/data/2.5",
  "description": "OpenWeatherMap API",
  "auth": {
    "type": "api_key",
    "api_key": "your-api-key",
    "api_key_param": "appid"
  },
  "endpoints": [
    {
      "name": "get_current_weather",
      "description": "Get current weather for a city",
      "method": "GET",
      "path": "/weather",
      "params": [
        {
          "name": "q",
          "type": "string",
          "location": "query",
          "required": true,
          "description": "City name"
        },
        {
          "name": "units",
          "type": "string",
          "location": "query",
          "required": false,
          "default": "metric",
          "enum": ["metric", "imperial", "kelvin"]
        }
      ]
    }
  ]
}
```

### Example 2: GitHub API

```json
{
  "name": "github",
  "base_url": "https://api.github.com",
  "description": "GitHub REST API",
  "auth": {
    "type": "bearer",
    "bearer_token": "ghp_your_token_here"
  },
  "headers": {
    "Accept": "application/vnd.github.v3+json"
  },
  "endpoints": [
    {
      "name": "get_user",
      "description": "Get a GitHub user's information",
      "method": "GET",
      "path": "/users/{username}",
      "params": [
        {
          "name": "username",
          "type": "string",
          "location": "path",
          "required": true,
          "description": "GitHub username"
        }
      ]
    }
  ]
}
```

## Authentication Types

### Bearer Token
```json
{
  "auth": {
    "type": "bearer",
    "bearer_token": "your-token-here"
  }
}
```

### API Key (Header)
```json
{
  "auth": {
    "type": "api_key",
    "api_key": "your-key-here",
    "api_key_header": "X-API-Key"
  }
}
```

### API Key (Query Parameter)
```json
{
  "auth": {
    "type": "api_key",
    "api_key": "your-key-here",
    "api_key_param": "api_key"
  }
}
```

### Basic Authentication
```json
{
  "auth": {
    "type": "basic",
    "username": "your-username",
    "password": "your-password"
  }
}
```

### Custom Headers
```json
{
  "auth": {
    "type": "custom",
    "custom_headers": {
      "X-Custom-Auth": "custom-value",
      "X-Client-ID": "client-123"
    }
  }
}
```

## Parameter Locations

- **query**: Query string parameters (`?param=value`)
- **path**: Path parameters (`/users/{id}`)
- **header**: HTTP headers
- **body**: Request body (for POST, PUT, PATCH)

## Parameter Types

- **string**: Text values
- **integer**: Whole numbers
- **number**: Decimal numbers
- **boolean**: true/false
- **array**: Lists of values
- **object**: JSON objects

## Advanced Features

### Custom Timeouts
```json
{
  "timeout": 60.0  // Timeout in seconds
}
```

### Enum Values
```json
{
  "name": "status",
  "type": "string",
  "enum": ["active", "inactive", "pending"]
}
```

### Default Values
```json
{
  "name": "page",
  "type": "integer",
  "default": 1
}
```

## Claude Desktop Configuration

### For Streamable HTTP Transport (Recommended)
```json
{
  "mcpServers": {
    "apiweaver": {
      "command": "apiweaver",
      "args": ["run", "--transport", "streamable-http", "--host", "127.0.0.1", "--port", "8000"]
    }
  }
}
```

### For STDIO Transport (Traditional)
```json
{
  "mcpServers": {
    "apiweaver": {
      "command": "apiweaver",
      "args": ["run"]
    }
  }
}
```

## Error Handling

The server provides detailed error messages for:
- Missing required parameters
- HTTP errors (with status codes)
- Connection failures
- Authentication errors
- Invalid configurations

## Tips

1. **Choose the Right Transport**: Use `streamable-http` for modern deployments, `stdio` for local tools
2. **Test First**: Always use `test_api_connection` after registering an API
3. **Start Simple**: Begin with GET endpoints before moving to complex POST requests
4. **Check Auth**: Ensure your authentication credentials are correct
5. **Use Descriptions**: Provide clear descriptions for better AI understanding
6. **Handle Errors**: The server will report HTTP errors with details

## Troubleshooting

### Common Issues

1. **401 Unauthorized**: Check your authentication credentials
2. **404 Not Found**: Verify the base URL and endpoint paths
3. **Timeout Errors**: Increase the timeout value for slow APIs
4. **SSL Errors**: Some APIs may require specific SSL configurations

### Debug Mode

Run with verbose logging (if installed):
```bash
apiweaver run --verbose
```

### Transport-Specific Issues

- **STDIO**: Ensure the client properly handles stdin/stdout communication
- **SSE**: Check that the HTTP endpoint is accessible and CORS is configured
- **Streamable HTTP**: Verify the MCP endpoint responds to HTTP requests

## Contributing

Feel free to extend this server with additional features:
- OAuth2 token refresh
- GraphQL support
- WebSocket endpoints
- Response caching
- Rate limiting
- Request retries

## License

MIT License - feel free to use and modify as needed.
