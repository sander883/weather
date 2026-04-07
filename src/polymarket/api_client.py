"""Polymarket API client for market data and trading."""

import requests
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import time
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OrderType(Enum):
    """Order types on Polymarket."""
    BUY = "BUY"
    SELL = "SELL"


class OutcomeType(Enum):
    """Outcome types."""
    YES = "YES"
    NO = "NO"


class PolymarketAPIClient:
    """Client for Polymarket REST API."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 base_url: str = "https://api.polymarket.com", timeout: int = 30):
        """
        Initialize Polymarket API client.

        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            base_url: Base URL for API
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self._setup_session()
        self.rate_limit_remaining = None
        self.rate_limit_reset = None

    def _setup_session(self) -> None:
        """Setup session with default headers."""
        self.session.headers.update({
            'User-Agent': 'PolymarketWeatherBot/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })

    def get_markets(self, limit: int = 100, offset: int = 0,
                   search_term: Optional[str] = None,
                   tag: Optional[str] = None,
                   status: str = 'active') -> Dict[str, Any]:
        """
        Get markets from Polymarket.

        Args:
            limit: Number of markets to return
            offset: Offset for pagination
            search_term: Search term for filtering
            tag: Tag filter (e.g., 'weather')
            status: Market status filter ('active', 'closed', 'resolved')

        Returns:
            Dictionary with markets list and metadata
        """
        try:
            params = {
                'limit': limit,
                'offset': offset,
                'status': status
            }

            if search_term:
                params['search'] = search_term
            if tag:
                params['tag'] = tag

            response = self._request('GET', '/markets', params=params)

            if response.get('status') == 'success':
                logger.info(f"Retrieved {len(response.get('data', []))} markets")
                return response
            else:
                logger.error(f"Failed to get markets: {response.get('error')}")
                return {'status': 'error', 'data': [], 'error': response.get('error')}

        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return {'status': 'error', 'data': [], 'error': str(e)}

    def get_market(self, market_id: str) -> Dict[str, Any]:
        """
        Get details for a specific market.

        Args:
            market_id: Polymarket market ID

        Returns:
            Market details dictionary
        """
        try:
            response = self._request('GET', f'/markets/{market_id}')

            if response.get('status') == 'success':
                logger.info(f"Retrieved market details for {market_id}")
                return response
            else:
                logger.error(f"Failed to get market {market_id}: {response.get('error')}")
                return {'status': 'error', 'error': response.get('error')}

        except Exception as e:
            logger.error(f"Error fetching market {market_id}: {e}")
            return {'status': 'error', 'error': str(e)}

    def get_market_prices(self, market_id: str) -> Dict[str, Any]:
        """
        Get current prices for a market.

        Args:
            market_id: Polymarket market ID

        Returns:
            Dictionary with yes/no prices
        """
        try:
            response = self._request('GET', f'/markets/{market_id}/prices')

            if response.get('status') == 'success':
                logger.debug(f"Retrieved prices for {market_id}")
                return response
            else:
                logger.error(f"Failed to get prices for {market_id}: {response.get('error')}")
                return {'status': 'error', 'error': response.get('error')}

        except Exception as e:
            logger.error(f"Error fetching prices for {market_id}: {e}")
            return {'status': 'error', 'error': str(e)}

    def place_order(self, market_id: str, outcome: str, amount: float,
                   price: float, order_type: str = 'BUY') -> Dict[str, Any]:
        """
        Place an order on a market.

        Args:
            market_id: Polymarket market ID
            outcome: 'YES' or 'NO'
            amount: Number of shares to buy
            price: Price per share
            order_type: 'BUY' or 'SELL'

        Returns:
            Order execution result
        """
        if not self.api_key:
            logger.error("API key required for order placement")
            return {'status': 'error', 'error': 'API key not configured'}

        try:
            payload = {
                'market_id': market_id,
                'outcome': outcome.upper(),
                'amount': amount,
                'price': price,
                'order_type': order_type.upper(),
                'timestamp': datetime.utcnow().isoformat()
            }

            response = self._request('POST', '/orders', json=payload)

            if response.get('status') == 'success':
                logger.info(f"Order placed: {response.get('data', {}).get('order_id')}")
                return response
            else:
                logger.error(f"Failed to place order: {response.get('error')}")
                return {'status': 'error', 'error': response.get('error')}

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'status': 'error', 'error': str(e)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation result
        """
        if not self.api_key:
            logger.error("API key required for order cancellation")
            return {'status': 'error', 'error': 'API key not configured'}

        try:
            response = self._request('DELETE', f'/orders/{order_id}')

            if response.get('status') == 'success':
                logger.info(f"Order {order_id} cancelled")
                return response
            else:
                logger.error(f"Failed to cancel order: {response.get('error')}")
                return {'status': 'error', 'error': response.get('error')}

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {'status': 'error', 'error': str(e)}

    def get_orders(self, status: str = 'open', market_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get user's orders.

        Args:
            status: Order status filter ('open', 'filled', 'cancelled')
            market_id: Optional market filter

        Returns:
            List of orders
        """
        if not self.api_key:
            logger.error("API key required to fetch orders")
            return {'status': 'error', 'data': [], 'error': 'API key not configured'}

        try:
            params = {'status': status}
            if market_id:
                params['market_id'] = market_id

            response = self._request('GET', '/orders', params=params)

            if response.get('status') == 'success':
                logger.info(f"Retrieved {len(response.get('data', []))} orders")
                return response
            else:
                logger.error(f"Failed to fetch orders: {response.get('error')}")
                return {'status': 'error', 'data': [], 'error': response.get('error')}

        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return {'status': 'error', 'data': [], 'error': str(e)}

    def get_portfolio(self) -> Dict[str, Any]:
        """
        Get user's portfolio.

        Returns:
            Portfolio with positions and balances
        """
        if not self.api_key:
            logger.error("API key required to fetch portfolio")
            return {'status': 'error', 'error': 'API key not configured'}

        try:
            response = self._request('GET', '/portfolio')

            if response.get('status') == 'success':
                logger.info("Retrieved portfolio")
                return response
            else:
                logger.error(f"Failed to fetch portfolio: {response.get('error')}")
                return {'status': 'error', 'error': response.get('error')}

        except Exception as e:
            logger.error(f"Error fetching portfolio: {e}")
            return {'status': 'error', 'error': str(e)}

    def get_balance(self) -> Dict[str, Any]:
        """
        Get user's account balance.

        Returns:
            Balance information
        """
        if not self.api_key:
            logger.error("API key required to fetch balance")
            return {'status': 'error', 'balance': 0, 'error': 'API key not configured'}

        try:
            response = self._request('GET', '/account/balance')

            if response.get('status') == 'success':
                balance = response.get('data', {}).get('balance', 0)
                logger.debug(f"Account balance: ${balance:.2f}")
                return response
            else:
                logger.error(f"Failed to fetch balance: {response.get('error')}")
                return {'status': 'error', 'balance': 0, 'error': response.get('error')}

        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {'status': 'error', 'balance': 0, 'error': str(e)}

    def search_markets(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """
        Search markets by keyword.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Search results
        """
        try:
            params = {
                'search': query,
                'limit': limit,
                'status': 'active'
            }

            response = self._request('GET', '/markets/search', params=params)

            if response.get('status') == 'success':
                logger.info(f"Found {len(response.get('data', []))} markets matching '{query}'")
                return response
            else:
                logger.error(f"Search failed: {response.get('error')}")
                return {'status': 'error', 'data': [], 'error': response.get('error')}

        except Exception as e:
            logger.error(f"Error searching markets: {e}")
            return {'status': 'error', 'data': [], 'error': str(e)}

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None,
                json: Optional[Dict] = None, retries: int = 3) -> Dict[str, Any]:
        """
        Make HTTP request to API with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json: JSON payload
            retries: Number of retries

        Returns:
            Response dictionary
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    timeout=self.timeout
                )

                # Update rate limit info
                if 'X-RateLimit-Remaining' in response.headers:
                    self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
                if 'X-RateLimit-Reset' in response.headers:
                    self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])

                # Check rate limiting
                if response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                return {
                    'status': 'success',
                    'data': response.json() if response.text else None,
                    'status_code': response.status_code
                }

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{retries}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return {'status': 'error', 'error': 'Request timeout'}

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}/{retries}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {'status': 'error', 'error': f'Connection error: {str(e)}'}

            except requests.exceptions.HTTPError as e:
                if response.status_code == 401:
                    logger.error("Authentication failed")
                    return {'status': 'error', 'error': 'Authentication failed'}
                elif response.status_code == 404:
                    logger.error(f"Resource not found: {endpoint}")
                    return {'status': 'error', 'error': 'Resource not found'}
                else:
                    logger.error(f"HTTP error: {e}")
                    return {'status': 'error', 'error': f'HTTP error: {str(e)}'}

            except Exception as e:
                logger.error(f"Request failed: {e}")
                return {'status': 'error', 'error': str(e)}

        return {'status': 'error', 'error': 'Max retries exceeded'}

    def close(self) -> None:
        """Close the session."""
        self.session.close()
        logger.info("Polymarket API client closed")
