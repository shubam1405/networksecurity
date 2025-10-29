pipeline {
    agent {
        docker {
            image 'python:3.10'
            args '-u root'   // run as root so you can install packages
        }
    }

    stages {
        stage('Install Dependencies') {
            steps {
                echo 'Installing dependencies...'
                sh '''
                python3 -m pip install --upgrade pip
                pip install -r requirements.txt || echo "requirements.txt not found, skipping..."
                '''
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running tests...'
                sh 'echo "Simulated test run complete"'
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deploying app...'
            }
        }
    }
}
