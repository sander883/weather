"""Unit tests for Polymarket API client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from src.polymarket.api_client import PolymarketAPIClient, OrderType, OutcomeType


@pytest.fixture
def api_client():
    """Create API client instance."""
    return PolymarketAPIClient(api_key='test_key', api_secret='test_secret')


@pytest.fixture
def mock_response():
    """Create mock response."""
    response = Mock()
    response.status_code = 200
    response.text = '{"data": []}'
    response.json.return_value = {"data": []}
    response.headers = {
        'X-RateLimit-Remaining': '100',
        'X-RateLimit-Reset': '1234567890'
    }
    return response


class TestPolymarketAPIClientInit:
    """Test API client initialization."""

    def test_init_with_credentials(self):
        """Test initialization with credentials."""
        client = PolymarketAPIClient(api_key='key', api_secret='secret')
        assert client.api_key == 'key'
        assert client.api_secret == 'secret'

    def test_init_without_credentials(self):
        """Test initialization without credentials."""
        client = PolymarketAPIClient()
        assert client.api_key is None
        assert client.api_secret is None

    def test_init_custom_base_url(self):
        """Test initialization with custom base URL."""
        url = 'https://custom.api.com'
        client = PolymarketAPIClient(base_url=url)
        assert client.base_url == url

    def test_session_setup(self, api_client):
        """Test that session is properly configured."""
        assert 'User-Agent' in api_client.session.headers
        assert 'Authorization' in api_client.session.headers


class TestGetMarkets:
    """Test get_markets method."""

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_get_markets_success(self, mock_request, api_client, mock_response):
        """Test successful market retrieval."""
        market_list = [
            {'id': 'market_1', 'question': 'Will it rain?'},
            {'id': 'market_2', 'question': 'Will it snow?'}
        ]
        mock_response.json.return_value = market_list
        mock_request.return_value = mock_response

        result = api_client.get_markets(limit=10)

        assert result['status'] == 'success'
        # The API returns the list directly, which becomes result['data']
        assert isinstance(result['data'], list)

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_get_markets_with_filters(self, mock_request, api_client, mock_response):
        """Test market retrieval with filters."""
        mock_request.return_value = mock_response

        api_client.get_markets(limit=20, search_term='rain', tag='weather')

        # Verify request parameters
        call_args = mock_request.call_args
        assert call_args[1]['params']['search'] == 'rain'
        assert call_args[1]['params']['tag'] == 'weather'
        assert call_args[1]['params']['limit'] == 20

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_get_markets_error(self, mock_request, api_client):
        """Test market retrieval error handling."""
        mock_request.side_effect = Exception('API error')

        result = api_client.get_markets()

        assert result['status'] == 'error'
        assert 'error' in result

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_get_markets_pagination(self, mock_request, api_client, mock_response):
        """Test pagination parameters."""
        mock_request.return_value = mock_response

        api_client.get_markets(limit=50, offset=100)

        call_args = mock_request.call_args
        assert call_args[1]['params']['limit'] == 50
        assert call_args[1]['params']['offset'] == 100


class TestGetMarket:
    """Test get_market method."""

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_get_market_success(self, mock_request, api_client, mock_response):
        """Test successful market details retrieval."""
        market_data = {
            'id': 'market_123',
            'question': 'Will it rain tomorrow?',
            'yes_price': 0.45,
            'no_price': 0.55
        }
        mock_response.json.return_value = market_data
        mock_request.return_value = mock_response

        result = api_client.get_market('market_123')

        assert result['status'] == 'success'
        assert result['data']['id'] == 'market_123'

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_get_market_not_found(self, mock_request, api_client):
        """Test market not found error."""
        response = Mock()
        response.status_code = 404
        response.raise_for_status.side_effect = requests.exceptions.HTTPError('404')
        mock_request.return_value = response

        result = api_client.get_market('nonexistent')

        assert result['status'] == 'error'


class TestGetMarketPrices:
    """Test get_market_prices method."""

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_get_prices_success(self, mock_request, api_client, mock_response):
        """Test successful price retrieval."""
        price_data = {
            'market_id': 'market_123',
            'yes_price': 0.45,
            'no_price': 0.55,
            'timestamp': '2024-01-01T00:00:00Z'
        }
        mock_response.json.return_value = price_data
        mock_request.return_value = mock_response

        result = api_client.get_market_prices('market_123')

        assert result['status'] == 'success'
        assert result['data']['yes_price'] == 0.45


class TestPlaceOrder:
    """Test place_order method."""

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_place_order_success(self, mock_request, api_client, mock_response):
        """Test successful order placement."""
        order_data = {
            'order_id': 'order_123',
            'market_id': 'market_123',
            'outcome': 'YES',
            'amount': 100,
            'price': 0.45,
            'status': 'FILLED'
        }
        mock_response.json.return_value = order_data
        mock_request.return_value = mock_response

        result = api_client.place_order(
            market_id='market_123',
            outcome='YES',
            amount=100,
            price=0.45
        )

        assert result['status'] == 'success'
        assert result['data']['order_id'] == 'order_123'

    def test_place_order_without_api_key(self):
        """Test order placement without API key."""
        client = PolymarketAPIClient()  # No API key

        result = client.place_order(
            market_id='market_123',
            outcome='YES',
            amount=100,
            price=0.45
        )

        assert result['status'] == 'error'
        assert 'API key' in result['error']

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_place_buy_yes_order(self, mock_request, api_client, mock_response):
        """Test buying YES shares."""
        mock_request.return_value = mock_response

        api_client.place_order(
            market_id='market_123',
            outcome='YES',
            amount=50,
            price=0.50
        )

        call_args = mock_request.call_args
        payload = call_args[1]['json']
        assert payload['outcome'] == 'YES'
        assert payload['amount'] == 50

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_place_buy_no_order(self, mock_request, api_client, mock_response):
        """Test buying NO shares."""
        mock_request.return_value = mock_response

        api_client.place_order(
            market_id='market_123',
            outcome='NO',
            amount=75,
            price=0.40
        )

        call_args = mock_request.call_args
        payload = call_args[1]['json']
        assert payload['outcome'] == 'NO'
        assert payload['amount'] == 75


class TestCancelOrder:
    """Test cancel_order method."""

    def test_cancel_order_without_api_key(self):
        """Test order cancellation without API key."""
        client = PolymarketAPIClient()

        result = client.cancel_order('order_123')

        assert result['status'] == 'error'

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_cancel_order_success(self, mock_request, api_client, mock_response):
        """Test successful order cancellation."""
        mock_response.json.return_value = {'order_id': 'order_123', 'status': 'CANCELLED'}
        mock_request.return_value = mock_response

        result = api_client.cancel_order('order_123')

        assert result['status'] == 'success'
        # Verify DELETE method was used (check in kwargs)
        call_args = mock_request.call_args
        assert call_args[1]['method'] == 'DELETE'


class TestGetOrders:
    """Test get_orders method."""

    def test_get_orders_without_api_key(self):
        """Test fetching orders without API key."""
        client = PolymarketAPIClient()

        result = client.get_orders()

        assert result['status'] == 'error'

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_get_orders_success(self, mock_request, api_client, mock_response):
        """Test successful orders retrieval."""
        orders = [
            {'order_id': 'order_1', 'status': 'FILLED'},
            {'order_id': 'order_2', 'status': 'OPEN'}
        ]
        mock_response.json.return_value = orders
        mock_request.return_value = mock_response

        result = api_client.get_orders(status='open')

        assert result['status'] == 'success'
        assert len(result['data']) == 2

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_get_orders_with_market_filter(self, mock_request, api_client, mock_response):
        """Test orders retrieval with market filter."""
        mock_request.return_value = mock_response

        api_client.get_orders(market_id='market_123')

        call_args = mock_request.call_args
        assert call_args[1]['params']['market_id'] == 'market_123'


class TestGetPortfolio:
    """Test get_portfolio method."""

    def test_get_portfolio_without_api_key(self):
        """Test portfolio retrieval without API key."""
        client = PolymarketAPIClient()

        result = client.get_portfolio()

        assert result['status'] == 'error'

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_get_portfolio_success(self, mock_request, api_client, mock_response):
        """Test successful portfolio retrieval."""
        portfolio = {
            'balance': 10000,
            'positions': [
                {'market_id': 'market_1', 'shares': 100}
            ]
        }
        mock_response.json.return_value = portfolio
        mock_request.return_value = mock_response

        result = api_client.get_portfolio()

        assert result['status'] == 'success'
        assert result['data']['balance'] == 10000


class TestGetBalance:
    """Test get_balance method."""

    def test_get_balance_without_api_key(self):
        """Test balance retrieval without API key."""
        client = PolymarketAPIClient()

        result = client.get_balance()

        assert result['status'] == 'error'
        assert result['balance'] == 0

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_get_balance_success(self, mock_request, api_client, mock_response):
        """Test successful balance retrieval."""
        mock_response.json.return_value = {'balance': 5000.50}
        mock_request.return_value = mock_response

        result = api_client.get_balance()

        assert result['status'] == 'success'
        assert result['data']['balance'] == 5000.50


class TestSearchMarkets:
    """Test search_markets method."""

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_search_markets_success(self, mock_request, api_client, mock_response):
        """Test successful market search."""
        markets = [
            {'id': 'market_1', 'question': 'Will it rain in NYC?'},
            {'id': 'market_2', 'question': 'Will it rain in LA?'}
        ]
        mock_response.json.return_value = markets
        mock_request.return_value = mock_response

        result = api_client.search_markets('rain')

        assert result['status'] == 'success'
        assert len(result['data']) == 2

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_search_markets_with_limit(self, mock_request, api_client, mock_response):
        """Test search with limit parameter."""
        mock_request.return_value = mock_response

        api_client.search_markets('weather', limit=50)

        call_args = mock_request.call_args
        assert call_args[1]['params']['search'] == 'weather'
        assert call_args[1]['params']['limit'] == 50


class TestRateLimiting:
    """Test rate limiting handling."""

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_rate_limit_headers(self, mock_request, api_client, mock_response):
        """Test rate limit header parsing."""
        mock_request.return_value = mock_response

        api_client.get_markets()

        assert api_client.rate_limit_remaining == 100
        assert api_client.rate_limit_reset == 1234567890

    @patch('src.polymarket.api_client.requests.Session.request')
    @patch('time.sleep')
    def test_rate_limit_429(self, mock_sleep, mock_request, api_client):
        """Test handling of 429 rate limit response."""
        response_429 = Mock()
        response_429.status_code = 429
        response_429.headers = {'Retry-After': '60'}

        response_200 = Mock()
        response_200.status_code = 200
        response_200.text = '{}'
        response_200.json.return_value = {}
        response_200.headers = {}

        mock_request.side_effect = [response_429, response_200]

        result = api_client.get_markets()

        assert result['status'] == 'success'
        mock_sleep.assert_called_once_with(60)


class TestErrorHandling:
    """Test error handling and retries."""

    @patch('src.polymarket.api_client.requests.Session.request')
    @patch('time.sleep')
    def test_connection_error_retry(self, mock_sleep, mock_request, api_client):
        """Test connection error retry logic."""
        response_ok = Mock()
        response_ok.status_code = 200
        response_ok.text = '{}'
        response_ok.json.return_value = {}
        response_ok.headers = {}

        mock_request.side_effect = [
            requests.exceptions.ConnectionError('Connection failed'),
            response_ok
        ]

        result = api_client.get_markets()

        assert result['status'] == 'success'
        assert mock_sleep.called

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_timeout_handling(self, mock_request, api_client):
        """Test timeout error handling."""
        mock_request.side_effect = requests.exceptions.Timeout('Request timeout')

        result = api_client.get_markets()

        assert result['status'] == 'error'
        assert 'timeout' in result['error'].lower()

    @patch('src.polymarket.api_client.requests.Session.request')
    def test_authentication_error(self, mock_request, api_client):
        """Test authentication error handling."""
        # Simulate 401 response that raises HTTPError
        mock_request.side_effect = requests.exceptions.HTTPError('401 Unauthorized')

        result = api_client.get_markets()

        assert result['status'] == 'error'
        assert 'error' in result


class TestEnums:
    """Test enum definitions."""

    def test_order_type_enum(self):
        """Test OrderType enum."""
        assert OrderType.BUY.value == 'BUY'
        assert OrderType.SELL.value == 'SELL'

    def test_outcome_type_enum(self):
        """Test OutcomeType enum."""
        assert OutcomeType.YES.value == 'YES'
        assert OutcomeType.NO.value == 'NO'
