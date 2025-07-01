import {
  Component,
  OnInit,
  ElementRef,
  OnDestroy,
  ViewChild,
  ChangeDetectorRef,
  PLATFORM_ID,
  Output,
  EventEmitter ,
  Inject
} from '@angular/core';
import { io, Socket } from 'socket.io-client';
import { isPlatformBrowser } from '@angular/common';
@Component({
  selector: 'app-sign-camera',
  imports: [],
  templateUrl: './sign-camera.component.html',
  styleUrl: './sign-camera.component.scss',
})
export class SignCameraComponent implements OnInit, OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  private stream: MediaStream | null = null;
  private _translatedText: string = 'Inicia la cámara para traducir...';
  private socket!: Socket;
  private canvas!: HTMLCanvasElement;
  private canvasContext!: CanvasRenderingContext2D | null;
  cameraEnabled: boolean = false;

  @Output() translationChanged = new EventEmitter<string>();

  get translatedText(): string {
    return this._translatedText;
  }

  set translatedText(value: string) {
    if (this._translatedText !== value) { // Only emit if value actually changed
      this._translatedText = value;
      this.translationChanged.emit(value); // Emit the new value
    }
  }

  // Inject PLATFORM_ID
  constructor(private cdRef: ChangeDetectorRef, @Inject(PLATFORM_ID) private platformId: Object) {}

  ngOnInit(): void {
    if (isPlatformBrowser(this.platformId)) { // Only connect to socket in browser
        this.socket = io('http://127.0.0.1:5000'); // Connect to your fast backend

        this.socket.on('connect', () => {
          console.log('Conectado al servidor de señas');
          this.translatedText = 'Cámara lista. Esperando señas...';
        });

        this.socket.on('response', (data: any) => {
          console.log('Server message:', data.data);
        });

        this.socket.on('prediction', (data: any) => {
          console.log('Predicción recibida:', data.text);
          this.translatedText = data.text;
        });

        this.socket.on('disconnect', () => {
          console.log('Desconectado del servidor de señas');
          this.translatedText = 'Desconectado del servidor.';
        });

        this.socket.on('connect_error', (error: any) => {
            console.error('Error de conexión a Socket.IO:', error);
            this.translatedText = 'Error al conectar con el servidor de traducción.'
        });
    }
  }

  ngAfterViewInit(): void {
    if (isPlatformBrowser(this.platformId)) { // Only initialize canvas in browser
      this.canvas = document.createElement('canvas'); // This line was causing the error
      this.canvasContext = this.canvas.getContext('2d');
    }
  }

  async startCamera(): Promise<void> {
    if (!isPlatformBrowser(this.platformId)) {
      console.warn('Cannot access camera outside of browser environment.');
      return;
    }

    // ... rest of your startCamera logic ...
    if (!navigator.mediaDevices?.getUserMedia) {
      console.error('La API de MediaDevices no es soportada por este navegador.');
      alert('Tu navegador no soporta el acceso a la cámara.');
      return;
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });

      this.cameraEnabled = true;
      this.cdRef.detectChanges();

      if (this.videoElement && this.videoElement.nativeElement) {
        this.videoElement.nativeElement.srcObject = this.stream;
        this.videoElement.nativeElement.play();

        this.videoElement.nativeElement.onloadedmetadata = () => {
          if (this.canvas && this.videoElement.nativeElement) { // Ensure canvas is initialized
            this.canvas.width = this.videoElement.nativeElement.videoWidth;
            this.canvas.height = this.videoElement.nativeElement.videoHeight;
            this.sendFrameToBackend();
          }
        };
      } else {
        console.error('videoElement is not available.');
      }
    } catch (err) {
      this.handleCameraError(err);
    }
  }

  private sendFrameToBackend(): void {
    if (!isPlatformBrowser(this.platformId)) {
      return; // Do not send frames if not in browser
    }
    // ... rest of your sendFrameToBackend logic ...
    if (this.videoElement && this.videoElement.nativeElement.readyState === 4 && this.canvasContext && this.socket) {
      this.canvasContext.drawImage(
        this.videoElement.nativeElement,
        0,
        0,
        this.canvas.width,
        this.canvas.height
      );

      const imageData = this.canvas.toDataURL('image/jpeg', 0.6).split(',')[1];
      this.socket.emit('video_frame', { image: imageData });
    }

    setTimeout(() => this.sendFrameToBackend(), 100);
  }

  stopCamera(): void {
    if (!isPlatformBrowser(this.platformId)) {
      return;
    }
    // ... rest of your stopCamera logic ...
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
      this.cameraEnabled = false;
      this.videoElement.nativeElement.srcObject = null;
      this.translatedText = 'Cámara detenida.';
    }
  }

  ngOnDestroy(): void {
    if (isPlatformBrowser(this.platformId)) {
      this.stopCamera();
      if (this.socket) {
        this.socket.disconnect();
      }
    }
  }

  // ... handleCameraError method remains the same ...
  private handleCameraError(error: any): void {
    console.error('Error al acceder a la cámara:', error);
    this.cameraEnabled = false;
    let errorMessage = 'No se pudo acceder a la cámara.';
    if (error.name === 'NotAllowedError') {
      errorMessage = 'Permiso denegado para acceder a la cámara. Por favor, otórgale permiso a tu navegador.';
    } else if (error.name === 'NotFoundError') {
      errorMessage = 'No se encontró ninguna cámara.';
    } else if (error.name === 'NotReadableError') {
      errorMessage = 'La cámara ya está en uso o hay un problema de hardware.';
    }
    alert(errorMessage);
  }
}
