0. The project uses Gradle to download the dependencies and setup the IDE files. It's like Maven just way better.

> ./gradlew build -x test

1. Setup the IDE files

> ./gradlew eclipse idea

2. Test file

See `BasicEsSparkTest`. Run it as a JUnit test - it will start spark on the fly and run the test. It will expect Elasticsearch to be running on the default host/port (localhost/9200)
