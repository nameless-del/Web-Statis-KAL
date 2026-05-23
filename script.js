document.addEventListener('DOMContentLoaded', () => {
    const listItems = document.querySelectorAll('.nav-items li');
    const iframe = document.getElementById('frame-tugas');

    // Mengganti warna background (class active) jika menu di klik
    listItems.forEach(item => {
        item.addEventListener('click', function() {
            listItems.forEach(li => li.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // Menghilangkan sidebar bawaan di dalam file tugas agar tidak terjadi double sidebar
    iframe.addEventListener('load', function() {
        try {
            // Mengakses dokumen di dalam iframe
            const iframeDocument = iframe.contentDocument || iframe.contentWindow.document;
            
            if (iframeDocument) {
                // Mencari elemen sidebar kiri di file asli dan menyembunyikannya
                const sidebarBawaan = iframeDocument.querySelector('.sidebar-left');
                if (sidebarBawaan) {
                    sidebarBawaan.style.display = 'none';
                }
            }
        } catch (error) {
            console.log("Catatan: Jika dijalankan murni di File Explorer, mungkin trik menyembunyikan sidebar akan diblokir oleh sistem keamanan browser (CORS). Sebaiknya buka file ini menggunakan Live Server di VS Code.");
        }
    });
});