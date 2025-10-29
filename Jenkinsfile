pipeline {
    agent any

    stages {
        stage('Checkout SCM') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing dependencies...'
                sh '''
                python3 -m pip install --upgrade pip
                pip3 install -r requirements.txt || echo "requirements.txt not found, skipping..."
                '''
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running tests...'
                sh '''
                if [ -f "test_mongodb.py" ]; then
                    python3 test_mongodb.py
                else
                    echo "No tests to run"
                fi
                '''
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deploying app...'
                sh 'echo "Simulated deploy stage success!"'
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check logs.'
        }
    }
}
