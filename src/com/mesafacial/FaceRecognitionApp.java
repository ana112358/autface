package com.mesafacial;

import org.opencv.core.*;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.nio.file.Paths;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import org.opencv.highgui.HighGui;

public class FaceRecognitionApp {

    private static final String DB_URL = "jdbc:sqlite:images_database.db";
    private static final String IMAGES_DIR = "images";
    private static final String OUTPUT_DIR = "output";

    private CascadeClassifier faceDetector;
    private FaceEmbeddingExtractor embeddingExtractor;

    public FaceRecognitionApp() {
        loadHaarCascade();
        loadEmbeddingModel();
        initializeDatabase();
    }

    private void loadHaarCascade() {
        String modelPath = "resources/models/haarcascade_frontalface_alt.xml";
        faceDetector = new CascadeClassifier(modelPath);

        if (faceDetector.empty()) {
            throw new RuntimeException("Failed to load Haar Cascade model from: " + modelPath);
        }
        System.out.println("Haar Cascade model loaded successfully from: " + modelPath);
    }

    private void loadEmbeddingModel() {
        String modelPath = "resources/models/dlib_face_recognition_resnet_model_v1.dat";
        embeddingExtractor = new FaceEmbeddingExtractor(modelPath);
    }

    private void initializeDatabase() {
        String createTableSQL = """
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    image_path TEXT NOT NULL UNIQUE,
                    features BLOB
                );
                """;

        try (Connection connection = DriverManager.getConnection(DB_URL);
             Statement statement = connection.createStatement()) {
            statement.execute(createTableSQL);
            System.out.println("Database initialized successfully.");
        } catch (SQLException e) {
            System.err.println("Error initializing database: " + e.getMessage());
        }
    }

    public void insertFace(String name, String imagePath, byte[] features) {
        String insertSQL = "INSERT OR IGNORE INTO faces (name, image_path, features) VALUES (?, ?, ?)";
        try (Connection connection = DriverManager.getConnection(DB_URL);
             PreparedStatement preparedStatement = connection.prepareStatement(insertSQL)) {

            preparedStatement.setString(1, name);
            preparedStatement.setString(2, imagePath);
            preparedStatement.setBytes(3, features);
            preparedStatement.executeUpdate();

            System.out.println("Face inserted: " + name);
        } catch (SQLException e) {
            System.err.println("Error inserting face: " + e.getMessage());
        }
    }

    public List<RegisteredFace> getRegisteredFaces() {
        List<RegisteredFace> faces = new ArrayList<>();
        String selectSQL = "SELECT name, features FROM faces";

        try (Connection connection = DriverManager.getConnection(DB_URL);
             Statement statement = connection.createStatement();
             ResultSet resultSet = statement.executeQuery(selectSQL)) {

            while (resultSet.next()) {
                String name = resultSet.getString("name");
                byte[] features = resultSet.getBytes("features");
                if (features != null) {
                    faces.add(new RegisteredFace(name, features));
                }
            }
        } catch (SQLException e) {
            System.err.println("Error retrieving faces: " + e.getMessage());
        }

        return faces;
    }

    public void processImages() {
        File folder = new File(IMAGES_DIR);

        if (!folder.exists() || !folder.isDirectory()) {
            throw new RuntimeException("Directory 'images' does not exist.");
        }

        List<RegisteredFace> registeredFaces = getRegisteredFaces();

        for (File file : folder.listFiles()) {
            if (file.isFile() && file.getName().endsWith(".jpg")) {
                processImage(file, registeredFaces);
            }
        }
    }

    private void processImage(File file, List<RegisteredFace> registeredFaces) {
        String imagePath = file.getAbsolutePath();
        System.out.println("Processing image: " + file.getName());

        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            System.err.println("Error loading image: " + imagePath);
            return;
        }

        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(image, faces);

        for (Rect faceRect : faces.toArray()) {
            Mat face = new Mat(image, faceRect);
            byte[] embedding = embeddingExtractor.extractEmbedding(face);

            if (embedding == null) {
                System.err.println("Failed to generate embedding for: " + file.getName());
                continue;
            }

            String matchedName = findMatchingFace(embedding, registeredFaces);
            Scalar rectangleColor = (matchedName != null) ? new Scalar(0, 255, 0) : new Scalar(0, 0, 255);

            Imgproc.rectangle(image, faceRect.tl(), faceRect.br(), rectangleColor, 2);
            if (matchedName != null) {
                System.out.println("Matched face: " + matchedName);
            } else {
                System.out.println("No match found.");
            }
        }

        saveOutputImage(image, file.getName());
    }

    private String findMatchingFace(byte[] embedding, List<RegisteredFace> registeredFaces) {
        for (RegisteredFace registeredFace : registeredFaces) {
            if (embeddingExtractor.compareEmbeddings(embedding, registeredFace.getFeatures())) {
                return registeredFace.getName();
            }
        }
        return null;
    }

    private void saveOutputImage(Mat image, String fileName) {
        File outputDir = new File(OUTPUT_DIR);
        if (!outputDir.exists()) {
            outputDir.mkdir();
        }

        String outputImagePath = OUTPUT_DIR + "/output_" + fileName;
        Imgcodecs.imwrite(outputImagePath, image);
        System.out.println("Output image saved: " + outputImagePath);
    }
    
    
 // Função para capturar imagem da câmera e salvar localmente
    public void captureAndSaveImage(String name) {
        VideoCapture camera = new VideoCapture(0); // Abrir webcam
        if (!camera.isOpened()) {
            System.out.println("Erro ao acessar a câmera.");
            return;
        }

        // Criar interface gráfica
        JFrame frame = new JFrame("Captura de Imagem");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(640, 480);
        frame.setLayout(new BorderLayout());

        JLabel cameraLabel = new JLabel();
        frame.add(cameraLabel, BorderLayout.CENTER);

        JButton captureButton = new JButton("Capturar");
        frame.add(captureButton, BorderLayout.SOUTH);

        frame.setVisible(true);

        // Thread para atualizar a imagem da câmera em tempo real
        Thread cameraThread = new Thread(() -> {
            Mat frameMat = new Mat();
            while (camera.isOpened()) {
                camera.read(frameMat);
                if (!frameMat.empty()) {
                    ImageIcon image = new ImageIcon(matToBufferedImage(frameMat));
                    cameraLabel.setIcon(image);
                }

                // Pequeno atraso para reduzir o uso da CPU
                try {
                    Thread.sleep(33); // ~30 FPS
                } catch (InterruptedException ignored) {}
            }
        });

        cameraThread.start();

        // Quando o botão for clicado, salva a imagem e fecha a câmera
        captureButton.addActionListener(e -> {
            Mat frameMat = new Mat();
            if (camera.read(frameMat)) {
                File directory = new File("images/");
                if (!directory.exists()) {
                    directory.mkdir();
                }
                String filePath = "images/" + name + ".jpg";
                Imgcodecs.imwrite(filePath, frameMat);
                JOptionPane.showMessageDialog(frame, "Imagem salva em: " + filePath);

                // Encerrar captura e fechar janela
                camera.release();
                frame.dispose();
            } else {
                JOptionPane.showMessageDialog(frame, "Erro ao capturar imagem!");
            }
        });
    }




    private BufferedImage matToBufferedImage(Mat matrix) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (matrix.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = matrix.channels() * matrix.cols() * matrix.rows();
        byte[] buffer = new byte[bufferSize];
        matrix.get(0, 0, buffer);
        BufferedImage image = new BufferedImage(matrix.cols(), matrix.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);
        return image;
    }




    public void initialize() {
        // Carrega a biblioteca OpenCV
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        // Inicializa o banco de dados SQLite
        initializeDatabase();
        
        // Carrega os modelos necessários (Agora chamados através da instância)
        loadHaarCascade();
        loadEmbeddingModel();
    }

    public void run() {
        Scanner scanner = new Scanner(System.in);
        int option = -1; 

        do {
            System.out.println("\nMenu de Reconhecimento Facial");
            System.out.println("1. Capturar e registrar rosto");
            System.out.println("2. Processar imagens para reconhecimento");
            System.out.println("3. Listar rostos registrados");
            System.out.println("4. Sair");
            System.out.print("Escolha uma opção: ");

            if (!scanner.hasNextInt()) { // Evita erro caso o usuário insira texto ao invés de número
                System.out.println("Opção inválida! Digite um número de 1 a 4.");
                scanner.next();
                continue;
            }

            option = scanner.nextInt();
            scanner.nextLine(); 

            switch (option) {
                case 1:
                    System.out.print("Digite o nome para registro: ");
                    String name = scanner.nextLine().trim();
                    if (!name.isEmpty()) {
                        captureAndSaveImage(name);
                    } else {
                        System.out.println("Nome inválido! O registro foi cancelado.");
                    }
                    break;
                case 2:
                    processImages();
                    break;
                case 3:
                    List<RegisteredFace> faces = getRegisteredFaces();
                    if (faces.isEmpty()) {
                        System.out.println("Nenhum rosto registrado.");
                    } else {
                        System.out.println("Rostos Registrados:");
                        for (RegisteredFace face : faces) {
                            System.out.println("- " + face.getName());
                        }
                    }
                    break;
                case 4:
                    System.out.println("Encerrando o programa...");
                    break;
                default:
                    System.out.println("Opção inválida! Tente novamente.");
            }
        } while (option != 4);

        scanner.close();
    }

    public static void main(String[] args) {
    	System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        FaceRecognitionApp app = new FaceRecognitionApp();
        app.initialize();
        app.run();
    }
}

class RegisteredFace {
    private final String name;
    private final byte[] features;

    public RegisteredFace(String name, byte[] features) {
        this.name = name;
        this.features = features;
    }

    public String getName() {
        return name;
    }

    public byte[] getFeatures() {
        return features;
    }
}

class FaceEmbeddingExtractor {
    public FaceEmbeddingExtractor(String modelPath) {
        System.out.println("Embedding model loaded from: " + modelPath);
    }

    public byte[] extractEmbedding(Mat face) {
        // Implemente aqui a lógica para extração de embeddings reais usando um modelo de reconhecimento facial.
        return new byte[128]; // Substitua isso pela extração real de embeddings.
    }

    public boolean compareEmbeddings(byte[] emb1, byte[] emb2) {
        if (emb1 == null || emb2 == null || emb1.length != emb2.length) {
            throw new IllegalArgumentException("Invalid embeddings.");
        }

        double distance = 0.0;
        for (int i = 0; i < emb1.length; i++) {
            double diff = emb1[i] - emb2[i];
            distance += diff * diff;
        }

        distance = Math.sqrt(distance);
        double threshold = 0.4; // Ajuste baseado em testes reais
        return distance < threshold;
    }
}
