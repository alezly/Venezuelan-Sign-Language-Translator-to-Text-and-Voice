import {
  Component,
  ElementRef,
  OnDestroy,
  ViewChild,
  ChangeDetectorRef,
} from '@angular/core';
@Component({
  selector: 'app-sign-camera',
  imports: [],
  templateUrl: './sign-camera.component.html',
  styleUrl: './sign-camera.component.scss',
})
export class SignCameraComponent {
  cameraEnabled = false;

  // Referencia al elemento <video> en el HTML.
  // Usamos un "setter" para actuar en cuanto el elemento esté disponible.
  @ViewChild('videoElement')
  set videoElement(el: ElementRef<HTMLVideoElement> | undefined) {
    if (el) {
      // Si el elemento <video> existe y tenemos un stream, lo asignamos.
      this._videoElement = el;
      if (this.stream) {
        this._videoElement.nativeElement.srcObject = this.stream;
      }
    }
  }

  private _videoElement: ElementRef<HTMLVideoElement> | undefined;
  private stream: MediaStream | null = null;

  constructor(private cdRef: ChangeDetectorRef) {}

  // Método público para ser llamado desde el HTML (ej: un botón)
  async startCamera(): Promise<void> {
    // Primero, comprueba si el navegador soporta la API
    if (!navigator.mediaDevices?.getUserMedia) {
      console.error(
        'La API de MediaDevices no es soportada por este navegador.'
      );
      alert('Tu navegador no soporta el acceso a la cámara.');
      return;
    }

    try {
      // Solicita permiso y acceso a la cámara de video
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }, // Pide video, no audio
        audio: false,
      });

      // Si tenemos éxito, actualizamos el estado
      this.cameraEnabled = true;

      // Forzamos la detección de cambios para que el @if se actualice
      // y el setter de videoElement se ejecute inmediatamente.
      this.cdRef.detectChanges();
    } catch (err) {
      this.handleCameraError(err);
    }
  }

  // Método para detener la cámara
  stopCamera(): void {
    if (this.stream) {
      // Detiene cada una de las pistas de video
      this.stream.getTracks().forEach((track) => track.stop());
    }
    this.stream = null;
    this.cameraEnabled = false;
  }

  // Manejador de errores
  private handleCameraError(error: any): void {
    console.error('Error al acceder a la cámara:', error);
    let message = 'Ocurrió un error al intentar acceder a la cámara.';
    if (
      error.name === 'NotAllowedError' ||
      error.name === 'PermissionDeniedError'
    ) {
      message =
        'Permiso para acceder a la cámara denegado. Por favor, habilita el acceso en la configuración de tu navegador.';
    } else if (
      error.name === 'NotFoundError' ||
      error.name === 'DevicesNotFoundError'
    ) {
      message = 'No se encontró una cámara conectada a tu dispositivo.';
    }
    alert(message);
    this.cameraEnabled = false;
  }

  // Hook del ciclo de vida: Se ejecuta cuando el componente es destruido
  ngOnDestroy(): void {
    // ¡MUY IMPORTANTE! Apaga la cámara para liberar recursos.
    // Si no haces esto, la luz de la cámara del usuario podría quedarse encendida.
    this.stopCamera();
  }
}
