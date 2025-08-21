web: gunicorn --bind 0.0.0.0:$PORT app:app
web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
