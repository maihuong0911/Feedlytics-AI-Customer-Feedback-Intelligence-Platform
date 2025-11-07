// Global Variables for DataTable instance
let dataTableInstance = null;

// Toggle Sidebar
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('mainContent');
    sidebar.classList.toggle('collapsed');
    mainContent.classList.toggle('expanded');
    if (window.innerWidth <= 992) {
        sidebar.classList.toggle('show');
    }
}

// Show Section
function showSection(sectionId) {
    document.querySelectorAll('.container-fluid.mt-5').forEach(section => {
        section.style.display = 'none';
    });
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.style.display = 'block';
    }
}

// Toggle Theme
function toggleTheme() {
    const body = document.body;
    const theme = body.getAttribute('data-bs-theme');
    const themeToggle = document.getElementById('themeToggle');
    if (theme === 'light') {
        body.setAttribute('data-bs-theme', 'dark');
        themeToggle.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
    } else {
        body.setAttribute('data-bs-theme', 'light');
        themeToggle.innerHTML = '<i class="fas fa-moon"></i> Dark Mode';
    }
}

// Global Search (Placeholder)
function globalSearch() {
    const searchTerm = document.getElementById('globalSearch').value.toLowerCase();
    console.log('Searching for:', searchTerm);
    // Logic tìm kiếm
}

// --- CORE FUNCTIONS (API INTEGRATION) ---

// 1. Train Model (Gửi yêu cầu tới API)
function trainModelWithSource() {
    const dataSource = document.getElementById('dataSource').value;
    const trainProgress = document.getElementById('trainProgress');
    const progressBar = trainProgress.querySelector('.progress-bar');
    const trainStatus = document.getElementById('trainStatus');

    trainProgress.style.display = 'block';
    let progress = 0;
    progressBar.style.width = '0%';
    progressBar.innerText = '0%';
    trainStatus.innerText = 'Đang gửi yêu cầu huấn luyện tới Back-end...';

    // Giả lập tiến trình UI trong khi chờ API
    const interval = setInterval(() => {
        progress += 5;
        if (progress < 95) {
            progressBar.style.width = `${progress}%`;
            progressBar.innerText = `${progress}%`;
        }
    }, 300); 

    fetch('/api/train_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data_source: dataSource })
    })
    .then(response => response.json())
    .then(data => {
        clearInterval(interval);
        progressBar.style.width = '100%';
        progressBar.innerText = '100%';
        trainProgress.style.display = 'none';
        
        if (data.status === 'success') {
            trainStatus.innerText = data.message;
            Swal.fire('Thành công', data.message, 'success');
        } else {
            trainStatus.innerText = `Lỗi: ${data.message}`;
            Swal.fire('Lỗi', data.message, 'error');
        }
    })
    .catch(error => {
        clearInterval(interval);
        trainProgress.style.display = 'none';
        trainStatus.innerText = 'Lỗi kết nối Back-end.';
        Swal.fire('Lỗi', 'Không thể kết nối đến máy chủ API.', 'error');
    });
}

// 2. Analyze Data File (Gửi file CSV tới API)
function analyzeDataFile() {
    const fileInput = document.getElementById('analyzeFile');
    const file = fileInput.files[0];

    if (!file) {
        Swal.fire('Lỗi', 'Vui lòng chọn một file CSV để phân tích!', 'error');
        return;
    }

    document.getElementById('loadingOverlay').style.display = 'flex';
    
    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/analyze_data', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loadingOverlay').style.display = 'none';
        
        if (data.status === 'success') {
            // Cập nhật giao diện
            updateMetrics(data.metrics);
            generateCharts(data.sentiments, data.ratings);
            updateDataTable(data.table_data);

            showSection('dashboard');
            Swal.fire('Thành công', `Phân tích ${file.name} hoàn tất!`, 'success');
        } else {
            Swal.fire('Lỗi', data.error || 'Lỗi không xác định khi phân tích.', 'error');
        }
    })
    .catch(error => {
        document.getElementById('loadingOverlay').style.display = 'none';
        Swal.fire('Lỗi', 'Lỗi kết nối hoặc xử lý Back-end. Kiểm tra console.', 'error');
        console.error('Fetch Error:', error);
    });
}

// 3. Analyze Single Text (Tạm thời không khả dụng trong bản này)
function analyzeSingleText() {
    Swal.fire('Thông báo', 'Chức năng phân tích văn bản đơn lẻ chưa được tích hợp API trong bản demo nhanh này. Vui lòng sử dụng chức năng Upload File CSV.', 'info');
}

// --- UI UPDATES ---

function updateMetrics(metrics) {
    document.getElementById('positiveMetric').innerText = metrics.positive_rate.toFixed(1) + '%';
    document.getElementById('negativeMetric').innerText = metrics.negative_rate.toFixed(1) + '%';
    document.getElementById('neutralMetric').innerText = metrics.neutral_rate.toFixed(1) + '%';
    document.getElementById('avgRatingMetric').innerText = metrics.avg_rating.toFixed(1);
}

function generateCharts(sentiments, ratings) {
    // 1. Pie Chart
    const counts = { positive: 0, negative: 0, neutral: 0 };
    sentiments.forEach(s => counts[s] = (counts[s] || 0) + 1);
    const pieData = [{
        values: [counts.positive, counts.negative, counts.neutral],
        labels: ['Positive', 'Negative', 'Neutral'],
        type: 'pie',
        marker: { colors: ['#28a745', '#dc3545', '#ffc107'] },
        hole: 0.4
    }];
    Plotly.newPlot('pieChart', pieData, { responsive: true, margin: { t: 0, b: 0, l: 0, r: 0 } });

    // 2. Histogram
    const histData = [{
        x: ratings.filter(r => r !== null && r !== undefined),
        type: 'histogram',
        marker: { color: 'rgb(55, 83, 109)' },
        xbins: { start: 0.5, end: 5.5, size: 1 } // Phân bố từ 1 đến 5
    }];
    Plotly.newPlot('histChart', histData, { 
        responsive: true, 
        xaxis: { title: 'Rating', tickvals: [1, 2, 3, 4, 5] }, 
        yaxis: { title: 'Số lượng' } 
    });
}

function updateDataTable(data) {
    // Hủy DataTables cũ và khởi tạo lại với dữ liệu mới
    if (dataTableInstance) {
        dataTableInstance.destroy();
        $('#dataTable').empty(); // Xóa thẻ header/body cũ
    }
    
    // Khởi tạo lại với dữ liệu mới
    dataTableInstance = $('#dataTable').DataTable({
        data: data,
        columns: [
            { title: "Text", data: "text" },
            { title: "Sentiment", data: "sentiment" },
            { title: "Rating", data: "rating" }
        ],
        responsive: true,
        destroy: true // Quan trọng để cho phép khởi tạo lại
    });
}


// --- UTILITY FUNCTIONS ---

function generateAllCharts() {
    // Tạm thời vô hiệu hóa chức năng này vì nó cần dữ liệu toàn bộ hệ thống
    Swal.fire('Thông báo', 'Chức năng này cần API tải toàn bộ dữ liệu hệ thống, vui lòng chạy phân tích file CSV trước.', 'info');
}

function exportChart(chartId) {
    const container = document.getElementById(chartId);
    if (!container || container.innerHTML === '') {
        Swal.fire('Lỗi', 'Biểu đồ trống. Vui lòng phân tích dữ liệu trước!', 'error');
        return;
    }
    
    document.getElementById('loadingOverlay').style.display = 'flex';
    Plotly.toImage(container, { format: 'png', height: 400, width: 600 })
        .then(function(url) {
            const link = document.createElement('a');
            link.download = `${chartId}.png`;
            link.href = url;
            link.click();
            document.getElementById('loadingOverlay').style.display = 'none';
            Swal.fire('Thành công', `Biểu đồ ${chartId} đã được xuất!`, 'success');
        })
        .catch(() => {
            document.getElementById('loadingOverlay').style.display = 'none';
            Swal.fire('Lỗi', 'Không thể xuất hình ảnh!', 'error');
        });
}

// Export PDF (Sử dụng html2canvas cho phần Dashboard)
function exportPDF() {
    document.getElementById('loadingOverlay').style.display = 'flex';
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    const dashboardElement = document.getElementById('dashboard');
    
    html2canvas(dashboardElement, { scale: 2 }).then(canvas => {
        const imgData = canvas.toDataURL('image/jpeg', 0.9);
        const imgWidth = 210; 
        const pageHeight = 295; 
        const imgHeight = canvas.height * imgWidth / canvas.width;
        let heightLeft = imgHeight;
        let position = 0;

        doc.addImage(imgData, 'JPEG', 0, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;

        // Thêm trang mới nếu cần (cho bảng dữ liệu lớn)
        while (heightLeft >= 0) {
            position = heightLeft - imgHeight;
            doc.addPage();
            doc.addImage(imgData, 'JPEG', 0, position, imgWidth, imgHeight);
            heightLeft -= pageHeight;
        }

        doc.save('feedback_report.pdf');
        document.getElementById('loadingOverlay').style.display = 'none';
        Swal.fire('Thành công', 'Báo cáo đã được xuất dưới dạng PDF!', 'success');
    }).catch((e) => {
        console.error("PDF Export Error:", e);
        document.getElementById('loadingOverlay').style.display = 'none';
        Swal.fire('Lỗi', 'Không thể xuất PDF!', 'error');
    });
}

// Clear All Data
function clearAllData() {
    Swal.fire({
        title: 'Bạn chắc chứ?',
        text: 'Hành động này sẽ xóa tất cả dữ liệu trên giao diện và không thể khôi phục!',
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: 'Xóa',
        cancelButtonText: 'Hủy'
    }).then((result) => {
        if (result.isConfirmed) {
            // Xóa dữ liệu trên UI
            updateMetrics({ positive_rate: 0, negative_rate: 0, neutral_rate: 0, avg_rating: 0 });
            Plotly.purge('pieChart');
            Plotly.purge('histChart');
            updateDataTable([]);
            Swal.fire('Đã Xóa!', 'Tất cả dữ liệu giao diện đã bị xóa.', 'success');
        }
    });
}

// Initialize DataTable and Charts
document.addEventListener('DOMContentLoaded', () => {
    showSection('dashboard');
    // Khởi tạo Dashboard với dữ liệu rỗng/0
    updateMetrics({ positive_rate: 0, negative_rate: 0, neutral_rate: 0, avg_rating: 0 });
    generateCharts([], []); 
    updateDataTable([]); 
});