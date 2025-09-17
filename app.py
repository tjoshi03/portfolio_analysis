from flask import Flask
from routes.portfolio_routes import portfolio_bp

app = Flask(__name__)
app.register_blueprint(portfolio_bp)

if __name__ == "__main__":
    app.run(debug=True)
