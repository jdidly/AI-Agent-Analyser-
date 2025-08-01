from flask import Flask, request, jsonify
import traceback
import logging
import config
from crypto_agent import run_simulation

app = Flask(__name__, static_folder="static")

@app.route('/api/run', methods=['POST'])
def api_run():
    try:
        params = request.json or {}
        # Allow user to override simulation params, else use defaults
        results = run_simulation(
            symbols=params.get('symbols'),
            profit_target=params.get('profit_target'),
            stop_loss=params.get('stop_loss'),
            confidence_threshold=params.get('confidence_threshold'),
            period=params.get('period'),
            initial_capital=params.get('initial_capital'),
            risk_pct=params.get('risk_pct'),
            spread=params.get('spread'),
            commission=params.get('commission')
        )
        # Convert DataFrame to dict for JSON
        return jsonify({
            'success': True,
            'results': results.to_dict(orient='records')
        })
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


# Serve the frontend
@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
