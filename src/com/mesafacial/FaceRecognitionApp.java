package com.mesafacial;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Point;

import java.io.File;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.Scanner;

public class FaceRecognitionApp {

    // Classe Camera
    static class Camera {
        private VideoCapture camera;

        public Camera() {
            camera = new VideoCapture(0);
        }

        public boolean estaAberta() {
            return camera.isOpened();
        }

        public void liberar() {
            camera.release();
        }

        public Mat capturarQuadro() {
            Mat quadro = new Mat();
            if (camera.isOpened()) {
                camera.read(quadro);
            }
            return quadro;
        }
    }

    // Classe DetectadorDeRosto
    static class DetectadorDeRosto {
        private CascadeClassifier detectorDeRosto;

        public DetectadorDeRosto(String caminhoModelo) {
            detectorDeRosto = new CascadeClassifier(caminhoModelo);
        }

        public MatOfRect detectarRostos(Mat quadro) {
            MatOfRect rostos = new MatOfRect();
            detectorDeRosto.detectMultiScale(quadro, rostos);
            return rostos;
        }
    }

    // Classe SalvadorDeImagem
    static class SalvadorDeImagem {

        public static void salvarImagem(Mat quadro, String nomeUsuario) {
            File diretorio = new File("imagens");
            if (!diretorio.exists()) {
                diretorio.mkdirs();
            }
            String caminhoArquivo = "imagens/" + nomeUsuario + ".jpg";
            Imgcodecs.imwrite(caminhoArquivo, quadro);
            System.out.println("Imagem salva como: " + new File(caminhoArquivo).getAbsolutePath());
        }

        public static String gerarHashDaImagem(Mat quadro) {
            try {
                byte[] imagemBytes = new byte[(int) (quadro.total() * quadro.channels())];
                quadro.get(0, 0, imagemBytes);

                // Gera o hash SHA-256
                MessageDigest digest = MessageDigest.getInstance("SHA-256");
                byte[] hashBytes = digest.digest(imagemBytes);

                // Converte para string hexadecimal
                StringBuilder stringHexadecimal = new StringBuilder();
                for (byte b : hashBytes) {
                    stringHexadecimal.append(String.format("%02x", b));
                }

                return stringHexadecimal.toString();
            } catch (NoSuchAlgorithmException e) {
                e.printStackTrace();
                return null;
            }
        }

        // Conectar ao banco de dados SQLite e salvar o nome e hash da imagem
        public static void salvarDadosDaImagemNoBancoDeDados(String nomeImagem, String hashImagem) {
            String url = "jdbc:sqlite:imagens_database.db";

            // Conectar ao banco de dados
            try (Connection conn = DriverManager.getConnection(url)) {
                if (conn != null) {
                    // Criar a tabela caso não exista
                    String criarTabelaSQL = "CREATE TABLE IF NOT EXISTS imagens (" +
                            "id INTEGER PRIMARY KEY AUTOINCREMENT," +
                            "nome TEXT NOT NULL," +
                            "hash TEXT NOT NULL);";
                    conn.createStatement().execute(criarTabelaSQL);

                    // Inserir dados da imagem
                    String inserirSQL = "INSERT INTO imagens (nome, hash) VALUES (?, ?)";
                    try (PreparedStatement pstmt = conn.prepareStatement(inserirSQL)) {
                        pstmt.setString(1, nomeImagem);
                        pstmt.setString(2, hashImagem);
                        pstmt.executeUpdate();
                        System.out.println("Dados da imagem salvos no banco de dados.");
                    }
                }
            } catch (SQLException e) {
                System.out.println(e.getMessage());
            }
        }
    }

    // Classe OuvinteDeCliqueDeMouse
    static class OuvinteDeCliqueDeMouse {
        private boolean capturaSolicitada = false;

        public boolean isCapturaSolicitada() {
            return capturaSolicitada;
        }

        public void resetarSolicitacaoCaptura() {
            capturaSolicitada = false;
        }

        public void setCapturaSolicitada(boolean capturaSolicitada) {
            this.capturaSolicitada = capturaSolicitada;
        }
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Solicita o nome do usuário antes de iniciar a câmera
        Scanner scanner = new Scanner(System.in);
        System.out.print("Digite seu nome para salvar a imagem: ");
        String nomeUsuario = scanner.nextLine().trim().replaceAll("\\s+", "_"); // Remove espaços
        scanner.close();

        // Inicialização das classes auxiliares
        Camera camera = new Camera();
        if (!camera.estaAberta()) {
            System.out.println("Erro ao abrir a câmera!");
            return;
        }

        DetectadorDeRosto detectadorDeRosto = new DetectadorDeRosto("resources/models/haarcascade_frontalface_alt.xml");
        OuvinteDeCliqueDeMouse ouvinteCliqueMouse = new OuvinteDeCliqueDeMouse();

        // Exibe o callback do mouse apenas uma vez, fora do loop de captura
        HighGui.setMouseCallback("Câmera - Clique no botão para capturar", (event, x, y, flags, userdata) -> {
            if (event == HighGui.EVENT_LBUTTONDOWN) {
                int larguraBotao = 200, alturaBotao = 50;
                Point topoEsquerdoBotao = new Point(camera.capturarQuadro().cols() - larguraBotao - 10, camera.capturarQuadro().rows() - alturaBotao - 10);
                Point inferiorDireitoBotao = new Point(camera.capturarQuadro().cols() - 10, camera.capturarQuadro().rows() - 10);

                if (x >= topoEsquerdoBotao.x && x <= inferiorDireitoBotao.x && y >= topoEsquerdoBotao.y && y <= inferiorDireitoBotao.y) {
                    ouvinteCliqueMouse.setCapturaSolicitada(true);
                }
            }
        });

        // Exibindo o loop da câmera e a captura ao clicar no botão
        while (true) {
            Mat quadro = camera.capturarQuadro();
            if (!quadro.empty()) {
                MatOfRect rostos = detectadorDeRosto.detectarRostos(quadro);

                for (Rect rect : rostos.toArray()) {
                    Imgproc.rectangle(quadro, rect.tl(), rect.br(), new Scalar(0, 255, 0), 3);
                }

                // Desenha o botão de captura na imagem
                int larguraBotao = 200, alturaBotao = 50;
                Point topoEsquerdoBotao = new Point(quadro.cols() - larguraBotao - 10, quadro.rows() - alturaBotao - 10);
                Point inferiorDireitoBotao = new Point(quadro.cols() - 10, quadro.rows() - 10);
                Imgproc.rectangle(quadro, topoEsquerdoBotao, inferiorDireitoBotao, new Scalar(255, 0, 0), -1); // Cor do botão
                Imgproc.putText(quadro, "Capturar", new Point(quadro.cols() - larguraBotao + 10, quadro.rows() - alturaBotao / 2),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 255, 255), 2); // Texto do botão

                // Exibe o frame com o botão de captura
                HighGui.imshow("Câmera - Clique no botão para capturar", quadro);

                // Verifica se o mouse clicou no botão
                int key = HighGui.waitKey(30);
                if (key == 27) { // Se pressionar "ESC", fecha a câmera
                    System.out.println("Saindo...");
                    break;
                }

                // Verifica o clique do mouse para capturar a imagem
                if (ouvinteCliqueMouse.isCapturaSolicitada()) {
                    // Captura a imagem
                    SalvadorDeImagem.salvarImagem(quadro, nomeUsuario);

                    // Gera o hash da imagem
                    String hashImagem = SalvadorDeImagem.gerarHashDaImagem(quadro);
                    System.out.println("Hash da imagem: " + hashImagem);

                    // Salva o nome e o hash no banco de dados
                    SalvadorDeImagem.salvarDadosDaImagemNoBancoDeDados(nomeUsuario + ".jpg", hashImagem);

                    ouvinteCliqueMouse.resetarSolicitacaoCaptura();
                    break; // Captura a imagem uma única vez
                }
            }
        }

        camera.liberar();
        HighGui.destroyAllWindows();
    }
}
