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
                sh 'npm install' // or your build command
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running tests...'
                sh 'npm test' // or your test command
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deploying project...'
                // your deploy command here
            }
        }
    }

    post {
        always {
            echo 'Cleaning up...'
        }
        success {
            echo '✅ Pipeline executed successfully!'
        }
        failure {
            echo '❌ Pipeline failed!'
        }
    }
}
