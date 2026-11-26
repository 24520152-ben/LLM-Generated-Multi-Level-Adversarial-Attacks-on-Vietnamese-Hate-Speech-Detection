# LLM-Generated Multi-Level Adversarial Attacks on Vietnamese Hate Speech Detection

*Tấn công đối kháng đa cấp độ bằng LLM đối với phát hiện ngôn từ thù ghét tiếng Việt*

> Cập nhật 26/11/2026. Đã có kết quả baseline trên dữ liệu gốc cho bốn model, mỗi model một run seed 42. Pipeline attack pilot đã được triển khai; chưa có kết quả attack hoặc annotation. Các thí nghiệm đối kháng bên dưới là kế hoạch cần kiểm chứng.

## Abstract

Các chỉnh sửa nhỏ hoặc cách diễn đạt khác có thể khiến mô hình phát hiện ngôn từ thù ghét dự đoán sai dù nội dung vẫn giữ nguyên. Nghiên cứu này sử dụng **một LLM với ba chế độ sinh ứng viên** ở cấp ký tự, từ/cụm từ và câu, kế thừa hướng prompting đa cấp độ của PromptAttack. Các chế độ dùng chung beam search, mục tiêu gây lỗi và giao thức kiểm định; điều kiện phối hợp được đánh giá dưới cùng ngân sách truy vấn classifier; chi phí sinh LLM được đo riêng. Thí nghiệm sử dụng ViHSD với ba nhãn `CLEAN`, `OFFENSIVE` và `HATE`. Kết quả baseline ban đầu, seed 42, đạt test macro-F1 từ 61,83% đến 66,67%; chưa đánh giá hiệu quả attack. Annotator độc lập kiểm tra khả năng đọc hiểu, giữ nghĩa, giữ nhãn và đúng cấp độ chỉnh sửa được yêu cầu. Nghiên cứu hướng tới cung cấp tập kiểm thử đối kháng được kiểm định và phân tích hiệu quả, mức tuân thủ từng chế độ, khả năng phối hợp và chuyển giao giữa mô hình.

## 1. Introduction

Phát hiện ngôn từ thù ghét trên mạng xã hội cần hoạt động ổn định khi người dùng thay đổi cách viết hoặc cách diễn đạt. Với tiếng Việt, từ nhiều âm tiết, ranh giới từ và hệ dấu tạo ra những yêu cầu riêng khi thích nghi attack văn bản. Một ứng viên trôi chảy hoặc có độ tương đồng ngữ nghĩa cao vẫn có thể làm đổi đối tượng bị công kích hoặc nhãn đúng.

Các phương pháp hiện có sử dụng nhiều cơ chế: TextBugger chỉnh sửa ký tự [2], TextFooler thay từ bằng tài nguyên từ vựng [3], BERT-Attack sinh thay thế theo ngữ cảnh [4], còn TREPAT khai thác bản viết lại từ mô hình ngôn ngữ [6]. PromptAttack [13] đã dùng LLM sinh attack ở cả ba cấp độ. Nghiên cứu này kế thừa cách điều khiển bằng prompt, giữ chung LLM và bộ tìm kiếm để khảo sát các chế độ chỉnh sửa trên Vietnamese hate speech detection. LLM sinh ứng viên được tách khỏi classifier đích, cho phép tấn công cả mô hình truyền thống và encoder.

Ba giả thuyết được đặt ra:

- **H1:** Với cùng LLM sinh ứng viên và beam search, hiệu quả của ba chế độ phụ thuộc vào mô hình đích; không có chế độ chiếm ưu thế trong mọi điều kiện khảo sát.
- **H2:** Phối hợp các cấp độ tăng tỷ lệ attack thành công hợp lệ so với từng cấp độ riêng dưới cùng ngân sách truy vấn classifier và quy tắc kiểm định hợp lệ; lượt gọi LLM và thời gian tìm kiếm được báo cáo riêng.
- **H3:** Kiểm định bởi con người làm thay đổi đáng kể ước lượng hiệu quả của ít nhất một điều kiện attack; khả năng đổi thứ hạng được phân tích bổ sung.

Đóng góp dự kiến gồm: (1) giao thức thích nghi và kiểm soát ba chế độ sinh attack bằng một LLM cho tiếng Việt; (2) tập kiểm thử đối kháng dẫn xuất từ ViHSD, có annotation và truy vết; (3) phân tích thực nghiệm kiểm định các giả thuyết trên. Prompting đa cấp độ và beam search được kế thừa; giá trị nghiên cứu cần được xác lập từ chất lượng tài nguyên và kết quả thực nghiệm.

## 2. Related Work

**Character-level attacks.** TextBugger [2] tìm các chỉnh sửa ký tự và từ gây lỗi thông qua phản hồi mô hình. Các thao tác ký tự là tài liệu tham khảo để thiết kế hướng dẫn và kiểm tra đầu ra LLM trong chế độ ký tự.

**Word/span-level attacks.** TextFooler [3] sử dụng ứng viên gần nghĩa; BERT-Attack [4] dùng masked language model để đề xuất thay thế theo ngữ cảnh. FBA [5] kết hợp thao tác từ với tìm kiếm Metropolis–Hastings. Các hướng này cung cấp cơ sở đối chiếu với thay thế từ theo ngữ cảnh do LLM sinh.

**LLM-based multi-level attacks.** PromptAttack của Xu và cộng sự [13], công bố tại ICLR 2024, dùng prompt gồm đầu vào gốc, mục tiêu và hướng dẫn chỉnh sửa ở cấp ký tự, từ và câu, kết hợp fidelity filtering. TREPAT [6] sử dụng bản viết lại từ mô hình ngôn ngữ và beam search. Chúng tôi kế thừa các ý tưởng này để đánh giá một LLM sinh ứng viên cho classifier độc lập, với kiểm định ba nhãn ViHSD và đối chứng cho lợi ích phối hợp cấp độ.

CoPA [14], EMNLP 2025 Main, dùng Qwen2.5-72B-Instruct làm paraphraser chính để tấn công bộ phát hiện văn bản máy sinh. Đây là tiền lệ cho LLM sinh bản viết lại, nhưng khác nhiệm vụ phát hiện ngôn từ thù ghét. TREPAT thử Llama-3.2-1B/3B-Instruct, Gemma-2-2B/9B-it, OLMo-7B-Instruct và Llama-3.1-8B-Instruct [6]; các model nhỏ cung cấp tiền lệ triển khai local. Lựa chọn Qwen3-8B cho đề tài là quyết định pilot, cần kiểm chứng trên tiếng Việt.

**Evaluation and Vietnamese NLP.** TextAttack [7] phân tách mục tiêu, ràng buộc, biến đổi và tìm kiếm; Morris và cộng sự [8] cho thấy cần kiểm định tính hợp lệ của adversarial examples. Lent [9] đã đưa tiếng Việt vào thử nghiệm đa ngôn ngữ với TextFooler trên phân loại chủ đề. Đề tài tập trung vào ViHSD [1], ba cấp độ chỉnh sửa và yêu cầu giữ nhãn của tác vụ, thay vì giả định adversarial NLP tiếng Việt chưa có tiền lệ.

## 3. Method

### 3.1. Threat Model and Objective

Thiết lập chính là **black-box, score-based, untargeted evasion**: attacker biết câu gốc và nhãn đánh giá, được truy vấn điểm số đầu ra nhưng không dùng gradient hoặc trọng số classifier. Mỗi đầu vào đi qua toàn bộ pipeline tiền xử lý và tokenizer của mô hình đích.

**Rubric nhãn.** Theo guideline ViHSD [1, Bảng 1], `CLEAN` (0) không có nội dung công kích/quấy rối; `OFFENSIVE` (1) có nội dung xúc phạm hoặc chửi tục nhưng không công kích đối tượng cụ thể; `HATE` (2) công kích một cá nhân hoặc một nhóm người, gồm cả công kích hàm ý/nghĩa bóng không có từ tục rõ ràng. `HATE` không chỉ giới hạn ở nhóm được bảo vệ. Prompt dùng rubric chung trong `VIHSD_LABEL_RUBRIC` tại `constants.py`; tiếng lóng, viết tắt và cách viết né tránh được xét theo ngữ cảnh, không tự suy diễn đối tượng hay bổ sung ngữ cảnh thiếu. Nhãn gốc là điều kiện cần bảo toàn; bộ sinh không gán nhãn lại hoặc sửa nghĩa để ép khớp rubric. Rubric hướng dẫn sinh, chưa xác nhận tính hợp lệ của ứng viên.

Với câu gốc $x$, nhãn $y$ và classifier $f$, một attack thành công hợp lệ phải thỏa:

$$
f(x)=y,\quad f(x')\ne y,\quad V(x,x',y)=1,
$$

với ngân sách tối đa $Q$ truy vấn classifier. Không đặt ngân sách lượt gọi LLM riêng; số lượt gọi và thời gian được ghi nhận như chi phí thực tế. $V$ được xét trong từng điều kiện attack, gồm khả năng đọc hiểu, giữ nghĩa, giữ nhãn và tuân thủ cấp độ chỉnh sửa được yêu cầu. Trong pilot, bộ lọc tự động chỉ kiểm tra định dạng và cấu trúc chỉnh sửa; chưa xác nhận $V$. Annotation quyết định tính hợp lệ cuối cùng. LLM chỉ đề xuất ứng viên; phản hồi từ classifier dẫn hướng beam search.

Gọi $s_c(x')$ là điểm đầu ra lớp $c$. Beam search ưu tiên ứng viên có margin lớn:

$$
g_f(x',y)=\max_{c\ne y}s_c(x')-s_y(x').
$$

Chỉ so sánh điểm trong cùng một classifier. SVM dùng decision score; các encoder dùng điểm đầu ra nhất quán với quy tắc dự đoán của chúng.

### 3.2. One LLM, Three Generation Modes

Với cùng checkpoint $\theta$, ba chế độ được biểu diễn bởi $G_\ell(z;x,y)=\mathrm{LLM}_\theta(P_{\mathrm{shared}},P_\ell,x,y,z)$, trong đó $z$ là câu đang mở rộng, $x,y$ là câu gốc và nhãn đánh giá, $\ell\in\{C,W,S\}$. Khung prompt chung quy định đầu ra, yêu cầu giữ nghĩa/nhãn và giới hạn chỉnh sửa; phần $P_\ell$ quy định cấp độ. Cung cấp câu gốc ở mỗi vòng để kiểm soát trôi nghĩa tích lũy.

| Chế độ | Hướng dẫn sinh ứng viên | Kiểm tra tuân thủ |
|---|---|---|
| **Ký tự ($G_C$)** | Tạo lỗi chèn, xóa, đảo hoặc thay ký tự tại số vị trí giới hạn; giữ nguyên phần còn lại | Chuẩn hóa NFC, Levenshtein trên grapheme cluster không quá 2 thao tác mỗi bước; annotator kiểm tra đúng loại chỉnh sửa |
| **Từ/cụm từ ($G_W$)** | Thay một số từ/cụm từ bằng cách diễn đạt gần nghĩa trong ngữ cảnh; giữ cấu trúc phần còn lại | Không quá 2 vùng diff; mỗi vùng trước/sau không quá 4 khối chữ/số Unicode; cho phép mở rộng/rút gọn cụm từ; annotator xác nhận cấp độ |
| **Câu ($G_S$)** | Paraphrase hoặc thay cấu trúc cú pháp, bảo toàn nội dung quyết định nhãn | Chỉ kiểm tra nội dung chữ/số thực sự thay đổi; không dùng số vùng diff để xác nhận S; annotator xác nhận paraphrase/cú pháp |

Pilot dùng `Qwen/Qwen3-8B`, lượng tử hóa 8-bit, tính toán FP16 và tắt thinking. Số ứng viên mặc định là 5/lượt, một lựa chọn triển khai cần kiểm tra trong pilot. Temperature 0.7, top-p 0.8 và top-k 20 theo khuyến nghị non-thinking của Qwen3 [15]. Model được chọn để thử nghiệm nhờ hỗ trợ tiếng Việt và khả năng triển khai trên T4; chưa có bằng chứng nó tối ưu cho ViHSD. Các lựa chọn này được kiểm tra trên validation rồi đóng băng checkpoint/revision, khung prompt, tham số giải mã và số ứng viên mỗi lượt gọi. Metadata chỉnh sửa được tính lại từ văn bản đầu ra. Kiểm tra tuân thủ chế độ được tách khỏi kiểm tra giữ nghĩa/nhãn; trường hợp cấp độ không rõ được xử lý theo guideline pilot và báo cáo riêng.

Bộ lọc pilot loại câu trùng lặp, hỏng định dạng hoặc không đạt giới hạn hình thức. C giới hạn 2 thao tác Levenshtein trên grapheme cluster mỗi bước. W giới hạn 2 vùng diff, mỗi vùng tối đa 4 đơn vị chữ/số ở cả trước và sau; đơn vị được tách bằng regex `[\p{L}\p{N}]+`, không phải từ tiếng Việt đã tách nghĩa. Diff có thể là thay, chèn hoặc xóa để không loại nhầm mở rộng/rút gọn cụm từ. Không còn loại W/S chỉ vì khoảng cách ký tự nhỏ; các bộ lọc có thể chồng lấn. W vẫn có thể nhận thay đổi sai nghĩa hoặc cả bình luận ngắn nếu nằm trong giới hạn. S chỉ yêu cầu có nội dung chữ/số thay đổi, bỏ điều kiện ít nhất hai vùng diff; việc vượt bộ lọc S không chứng minh paraphrase hay đổi cú pháp. W/S loại thay đổi chỉ ở khoảng trắng, dấu câu hoặc chữ hoa/thường. Pilot chưa có bộ lọc ngữ nghĩa tự động; annotation xác nhận cấp độ thực tế, giữ nghĩa và giữ nhãn. Đầu ra không đạt hoặc từ chối sinh được ghi nhận cùng lượt gọi sinh. Annotator xác nhận chất lượng cuối cùng thay vì dùng chính LLM sinh để tự chứng nhận tính hợp lệ.

### 3.3. Unified Beam Search

Khởi tạo beam bằng câu gốc. Ở mỗi vòng, bộ sinh được phép tạo ứng viên từ các câu trong beam; hệ thống lọc, loại trùng, truy vấn classifier và giữ tối đa $b$ câu có margin tốt nhất để mở rộng. Các câu gây lỗi được lưu trong candidate pool. Tìm kiếm dừng khi hết ngân sách hoặc một vòng không có ứng viên mới được chấm. Không dừng ở lần đổi nhãn đầu tiên để có thể kiểm định các ứng viên gây lỗi tiếp theo.

Bốn điều kiện dùng cùng bộ tìm kiếm là **C**, **W**, **S** và **MIX**. MIX cho phép một chuỗi chỉnh sửa đi qua nhiều chế độ của cùng LLM; tổng ngân sách $Q$ vẫn bằng một lần attack đơn cấp. Mỗi parent được mở rộng bằng cả ba chế độ, với thứ tự bắt đầu luân phiên theo câu và vòng, độc lập với Q; kiểm tra chế độ xét bước chỉnh sửa hiện tại. Chi phí LLM có thể khác nhau giữa các điều kiện và được báo cáo riêng.

Đánh giá câu gốc để xác định tập đủ điều kiện là bước chuẩn bị riêng, ngoài ngân sách attack. Mọi đánh giá classifier trong quá trình tìm kiếm đều được tính vào ngân sách; một batch chứa $n$ đầu vào tương ứng $n$ đánh giá. Riêng kiểm tra tuân thủ chế độ ký tự dùng số thao tác Levenshtein trên grapheme cluster của **bước hiện tại**; thí nghiệm không tính hay lưu khoảng cách giữa câu gốc và biến thể. Lịch sử các chế độ chỉnh sửa được lưu để truy vết chuỗi MIX.

## 4. Adversarial Test Set Construction

Dữ liệu nguồn là test set ViHSD [1], giữ ba nhãn theo hướng dẫn corpus. Pilot và phát triển phương pháp sử dụng validation. ID câu test, quy mô lấy mẫu, ngân sách và annotation guideline được khóa trước thí nghiệm chính. Nếu lấy mẫu, sử dụng quy tắc phân tầng theo nhãn, độc lập với kết quả attack. Câu gốc được kiểm định trước khi sinh attack; nhãn nguồn và nhãn sau phân xử được lưu riêng.

Ít nhất hai annotator thực hiện hai bước: (1) gán nhãn câu biến thể độc lập, không thấy nhãn nguồn hoặc dự đoán mô hình; (2) đánh giá cặp câu về khả năng đọc hiểu, giữ nghĩa, đối tượng, nội dung công kích và cấp độ chỉnh sửa thực tế. Với chuỗi MIX, kiểm tra từng bước bằng query cha thay vì chỉ nhìn hai đầu chuỗi. Người thứ ba phân xử bất đồng. Báo cáo đồng thuận trước phân xử, dự kiến dùng Krippendorff's alpha và tỷ lệ đồng thuận.

Một cặp chỉ được chấp nhận khi giữ nghĩa, giữ nhãn và đáp ứng cấp độ chỉnh sửa yêu cầu; việc qua bộ lọc tự động chưa đủ để xác nhận hợp lệ. Các ứng viên dùng để tuyên bố thành công hợp lệ phải được kiểm định; để kết luận một lần attack không có thành công hợp lệ, cần kiểm tra hết candidate pool tương ứng. Nếu nguồn lực hạn chế, xác định trước một tập đánh giá chính được kiểm định đầy đủ. Không loại mẫu chưa kiểm định khỏi mẫu số rồi diễn giải như ASR hợp lệ toàn test.

Trong bản phát hành cuối cùng, mỗi bản ghi dự kiến lưu ID gốc, nhãn, câu đối kháng, checkpoint/prompt/chế độ LLM, classifier/seed, lịch sử chỉnh sửa, số truy vấn, chi phí sinh, kết quả tuân thủ và annotation. Các biến thể của cùng câu giữ nguyên nhóm nguồn và split. Tập challenge set gồm các mẫu thành công được phát hành kèm thông tin mô hình sinh và nhật ký thống kê thành công/thất bại. Hai CSV pilot hiện tại được mô tả ở Mục 6.2; chưa có annotation hoặc metadata phát hành đầy đủ.

## 5. Experimental Setup

**Victim models.** Đã huấn luyện character TF-IDF + Linear SVM, PhoBERT-base [10], ViSoBERT [11] và XLM-R-base [12] với seed 42. Ba encoder chạy đủ 10 epoch, chọn checkpoint bằng macro-F1 validation cao nhất rồi đánh giá test. Cấu hình hiện tại: AdamW, learning rate $2\times10^{-5}$, batch size 16, max length 256, weight decay 0.01 và gradient clipping 1.0. PhoBERT tách từ bằng Underthesea; ViSoBERT và XLM-R nhận văn bản gốc. SVM dùng character TF-IDF n-gram 2–5, min-df 1, LinearSVC với C=1 và class-weight balanced. Các thiết lập khác dùng mặc định thư viện. Mục tiêu đánh giá cuối cùng là ba seed cho các mô hình có tính ngẫu nhiên; hiện mới hoàn thành một seed. Toàn bộ tiền xử lý được giữ nhất quán trước/sau attack.

**Comparisons.** Thí nghiệm chính so sánh C/W/S/MIX. Baseline sinh độc lập theo PromptAttack dùng cùng LLM và hướng dẫn cấp độ, sinh từ câu gốc mà không mở rộng qua beam; đây là bản thích nghi cho classifier độc lập, không đồng nhất với thiết lập LLM tự tấn công trong bài gốc. Đối chứng bổ sung gồm beam width $b=1$ và ba nhánh đơn cấp chạy riêng với cùng tổng $Q$ nhưng không nối tiếp chỉnh sửa. Đối chứng cuối giúp phân biệt lợi ích thử nhiều chế độ với lợi ích phối hợp trong một chuỗi.

**Budgets and reproducibility.** Khảo sát trên validation với $Q \in \{10,50,100\}$ và $b \in \{1,3,5\}$. Mốc Q và b=5 dựa trên TREPAT; b=1 và b=3 là đối chứng pilot. Với mỗi câu/chế độ/b, chỉ chạy một quỹ đạo đến Q=100 rồi đọc kết quả ở các prefix 10/50/100. Prompt, seed từng lượt và thứ tự sinh/chấm không phụ thuộc Q. Chọn cấu hình chung trên validation, công bố cùng checkpoint LLM, prompt, seed và quy tắc dừng. Giữ chung giới hạn token đầu ra mỗi lượt gọi; báo cáo lượt gọi LLM, thời gian, số đầu ra bị loại và chi phí sinh lại. Dùng cùng LLM không đồng nghĩa ba chế độ có cùng chi phí thực tế. Loại trùng không tiêu tốn query mới; chi phí bộ lọc hiện nằm trong thời gian tìm kiếm tổng. Không điều chỉnh phương pháp theo kết quả test.

**Cấu hình pilot.** Dùng Qwen3-8B 8-bit trên `cuda:0` và encoder classifier trên `cuda:1` của Kaggle T4 ×2, còn SVM dùng CPU, chỉ inference. Mặc định seed 42, tối đa 2.048 token đầu ra/lượt. Bốn notebook `scripts/pilot_{svm,phobert,visobert,xlmr}.ipynb` nạp checkpoint tương ứng trực tiếp từ Kaggle Dataset đã gắn vào notebook. Mỗi classifier lấy 30 câu được dự đoán đúng mỗi nhãn (90 câu/model), lấy mẫu cố định trên validation bằng seed 42. Trong từng classifier, cùng tập câu được dùng cho mọi chế độ và beam width; ID giữa các classifier có thể khác nhau do điều kiện dự đoán đúng. Các notebook chạy lần lượt trong cùng một phiên GPU. Pilot dùng để kiểm tra chất lượng ứng viên và chọn cấu hình, chưa phải đánh giá test cuối cùng. Ngưỡng cấu trúc C/W, số ứng viên và số token đầu ra là lựa chọn triển khai; đặc biệt `max_span_tokens=4` là mặc định pilot cần kiểm tra trên validation, chưa phải ngưỡng đã được xác lập cho tiếng Việt. Chỉ sampling non-thinking có nguồn từ Qwen3 [15].

Từng beam width chạy một lần tới Q=100; seed và thứ tự sinh/chấm không phụ thuộc ngân sách. Trạng thái ở Q=10/50 là prefix của cùng lượt chạy; không chọn riêng các lần chạy có kết quả thuận lợi. Nếu hết ứng viên trước một mốc, giữ kết quả và số query thực dùng. Khi chọn cấu hình, xét ASR sau kiểm định cùng chi phí; không chọn Q chỉ vì ASR tự động cao nhất. Thời gian toàn quỹ đạo chỉ ghi một lần ở dòng ngân sách lớn nhất (cột `seconds`); các mốc nhỏ hơn để trống. Số lượt gọi LLM vẫn được ghi theo từng prefix. Không lưu thống kê token riêng trong pilot.

**Transfer setting.** Sinh trên mô hình A, khóa câu đối kháng rồi đánh giá trên mô hình B không tham gia tìm kiếm. Báo cáo khả năng chuyển giao cùng số lượng và thành phần nhãn của tập đủ điều kiện.

## 6. Preliminary Results and Implementation Status

### 6.1. Clean Baselines

Các giá trị dưới đây là phần trăm, lấy trực tiếp từ `metrics.csv` của từng run; validation là epoch có macro-F1 cao nhất. Tất cả là dữ liệu gốc chưa bị attack, mỗi model một run seed 42. SVM không có epoch.

| Model | Epoch được chọn | Val macro-F1 | Val accuracy | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|---:|---:|
| Linear SVM | — | 62.59 | 86.34 | 61.83 | 86.72 |
| PhoBERT-base | 3 | 67.46 | 87.84 | 64.10 | 87.53 |
| ViSoBERT | 1 | 68.33 | 87.24 | 66.67 | 87.46 |
| XLM-R-base | 5 | 66.22 | 85.67 | 64.90 | 85.72 |

ViSoBERT có validation macro-F1 cao nhất (68,33%), có thể được chạy trước trong pilot bốn classifier dựa trên validation. Test macro-F1 của run này là 66,67%, cao hơn SVM 4,84 điểm phần trăm; PhoBERT có test accuracy cao nhất (87,53%). Các chênh lệch này là quan sát từ một seed, chưa có khoảng tin cậy hoặc kiểm định ý nghĩa thống kê và chưa cho biết độ bền trước attack.

Nguồn số liệu là `artifacts/baselines/<model>/metrics.csv`, với `<model>` lần lượt là `svm`, `phobert`, `visobert` và `xlmr`. Checkpoint tương ứng ở `artifacts/baselines/<model>/checkpoint/`. Mỗi model lưu trực tiếp `checkpoint/` và `metrics.csv`, không thêm thư mục seed hoặc thời gian. CSV lưu cả 10 epoch validation của encoder cùng kết quả test của checkpoint được chọn. Chạy lại model sẽ thay checkpoint và metrics tại cùng đường dẫn; muốn giữ một thí nghiệm khác có thể chỉ định `artifact_dir` riêng. Artifacts là dữ liệu cục bộ bị Git bỏ qua; bảng kết quả trong proposal và README được theo dõi cùng source.

### 6.2. Implementation and Next Experiments

Đã triển khai bốn notebook `scripts/pilot_{svm,phobert,visobert,xlmr}.ipynb`, nạp checkpoint baseline, lọc câu validation dự đoán đúng, sinh/lọc ứng viên C/W/S/MIX, beam search và grid dùng chung quỹ đạo query. Cấu hình baseline ở `src/vihsd_attack/classifiers/config.py`, cấu hình attack ở `src/vihsd_attack/attacks/config.py`; nhãn và thư mục artifacts dùng chung ở `src/vihsd_attack/config.py`, prompt ở `src/vihsd_attack/constants.py`. Notebook clone source và chạy qua môi trường Python 3.12 được khóa bằng `uv.lock`.

Pipeline lưu trực tiếp hai CSV tối giản. `artifacts/attacks/<model>/candidates.csv` có 9 cột theo thứ tự: `id`, `mode`, `beam_width`, `query`, `parent_query`, `step_mode`, `candidate`, `prediction`, `status`. Margin chỉ dùng trong bộ nhớ để xếp hạng beam, có thể chấm lại từ checkpoint khi cần; không lưu margin hoặc số lượt sinh trên từng ứng viên. Query câu cha bằng 0 trỏ tới câu gốc; các tham chiếu này dựng lại chuỗi chỉnh sửa mà không lặp nội dung câu cha hoặc lưu path riêng. Giữ cả ứng viên bị lọc để rà false rejection trong pilot; phản hồi không parse được chỉ lưu một lần trong cột ứng viên với trạng thái `invalid_output`. `artifacts/attacks/<model>/results.csv` có 10 cột: ID, chế độ, beam width, câu gốc, nhãn gốc, ngân sách, query đã dùng, thành công tự động, số lượt gọi LLM và thời gian. File kết quả giữ cả thất bại; câu gốc/nhãn chỉ nằm ở file này, ghép theo ID khi kiểm định ứng viên. Không lặp ứng viên/dự đoán trong file kết quả; thời gian chỉ xuất hiện ở dòng ngân sách lớn nhất. Mỗi notebook chỉ định `output_dir` riêng theo model để không ghi đè kết quả model khác. Không thêm thư mục seed hoặc thời gian; chạy lại pilot của cùng model thay hai CSV tại cùng đường dẫn. Gọi `run_pilot()` không có `output_dir` vẫn mặc định lưu vào `artifacts/attacks/`. Đã kiểm tra logic đếm query, prefix ngân sách và dừng tìm kiếm bằng 18 test CPU; đã thử tokenizer và API sinh Qwen3 bằng một model nhỏ khởi tạo ngẫu nhiên. Các kiểm tra kỹ thuật này không phải kết quả nghiên cứu; chưa chạy/đo inference Qwen3-8B 8-bit trên T4, chưa có CSV attack hoặc annotation tại thời điểm cập nhật.

Bước tiếp theo là chạy kiểm tra trên Kaggle, thực hiện pilot validation và annotation, cố định một cấu hình chung cho C/W/S/MIX rồi đánh giá test. Đối chứng sinh độc lập, ba nhánh riêng, transfer và đánh giá nhiều seed vẫn là thí nghiệm dự kiến.

## 7. Evaluation and Planned Analysis

Với mỗi mô hình, $E_f$ là tập câu gốc được dự đoán đúng. Mọi điều kiện attack sử dụng cùng $E_f$; trường hợp không có ứng viên hoặc hết ngân sách vẫn nằm trong mẫu số. Gọi $A_i$ là candidate pool của câu $i$:

$$
\mathrm{ASR}_{\mathrm{valid}}=
\frac{\sum_{i\in E_f}\mathbf{1}[\exists x'\in A_i:f(x')\ne y_i\land V(x_i,x',y_i)=1]}{|E_f|}.
$$

Báo cáo **ASR tự động** trên ứng viên qua bộ lọc cấu trúc và **ASR hợp lệ** sau annotation, kèm tỷ lệ ứng viên gây lỗi được xác nhận hợp lệ. ASR tự động chưa chứng minh giữ nghĩa/nhãn hoặc đúng cấp độ chỉnh sửa. ASR hợp lệ yêu cầu annotation xác nhận cả ba tiêu chí. **Tỷ lệ tuân thủ chế độ** là số ứng viên đạt yêu cầu cấp độ chia cho tổng số vị trí ứng viên được yêu cầu qua các lượt gọi; vị trí thiếu, không parse được hoặc bị từ chối được tính không đạt. Tỷ lệ từ chối, trùng lặp và mức đạt trên các đầu ra parse được được báo cáo riêng. Chỉ số bổ sung gồm truy vấn classifier, lượt gọi LLM, thời gian và ASR theo nhãn và chuyển đổi `HATE/OFFENSIVE → CLEAN`.

Để đo hiệu năng toàn test, chọn ứng viên hợp lệ theo thứ tự ưu tiên đã khóa; nếu không có thì giữ câu gốc. Các câu sai sẵn ở bản gốc được giữ nguyên. Báo cáo macro-F1 và F1 theo lớp trên cùng tập trước/sau attack. Điểm trên challenge set chỉ gồm các thành công được chọn được trình bày riêng với ASR của quá trình tìm kiếm.

Phân tích dự kiến gồm: (1) ASR theo ngân sách và tỷ lệ tuân thủ chế độ cho H1; (2) MIX so với từng cấp độ và đối chứng ba nhánh riêng cho H2; (3) chênh lệch trước/sau annotation cho H3; (4) ma trận chuyển giao và lỗi LLM ngoài phạm vi hướng dẫn. So sánh giữa mô hình bổ sung kết quả trên tập giao các câu cùng dự đoán đúng, kèm độ bao phủ. Ứng viên sai chế độ không được chuyển sang nhóm khác để tính thành công của điều kiện đã yêu cầu.

Dùng paired bootstrap theo **ID câu gốc** để ước lượng khoảng tin cậy, báo cáo riêng biến thiên giữa seed. Các so sánh chính và ngưỡng chênh lệch thực tiễn được chốt trước test; điều chỉnh kiểm định nhiều so sánh khi cần. Sự thay đổi tách từ và phân mảnh subword được xem là phân tích thăm dò.

## 8. Conclusion

Đã hoàn thành bốn baseline seed 42 và triển khai pipeline pilot dùng Qwen3-8B 8-bit, ba chế độ sinh ứng viên và beam search. Pilot được chuẩn bị cho cả bốn classifier; ViSoBERT có macro-F1 validation cao nhất trong các run hiện có nên có thể được chạy trước. Hiệu quả attack, mức tuân thủ cấp độ và lợi ích phối hợp vẫn cần được đánh giá trên validation, cố định cấu hình rồi kiểm chứng trên test với annotation. Chưa có kết luận về H1–H3.

## Limitations

Kết quả baseline hiện chỉ có một seed; chưa thể suy luận độ ổn định hoặc ý nghĩa thống kê. Bộ lọc W dùng diff trên đơn vị chữ/số, nhạy với căn chỉnh và có thể nhận lỗi ký tự hoặc viết lại quá rộng đối với bình luận ngắn. Bộ lọc S chỉ kiểm tra hình thức, có thể nhận thay từ hoặc lỗi chính tả; S chưa được xác nhận cho tới annotation. Các cấp độ không được giả định tách biệt tuyệt đối; cần kiểm tra cả nhóm qua và bị loại trong annotation validation trước khi cố định giao thức. Kết luận bị giới hạn bởi ViHSD, các classifier và checkpoint LLM được chọn. Cùng LLM và beam search vẫn có thể khác nhau về khả năng tuân thủ, chất lượng và chi phí giữa các chế độ; kết quả không tự động khái quát sang LLM khác. Không đo khoảng cách tổng nên không thể kết luận các điều kiện có cùng mức thay đổi văn bản; lợi ích MIX cần được diễn giải cùng với lịch sử chỉnh sửa và kiểm định bởi con người. Ranh giới giữa thay từ và paraphrase có trường hợp mơ hồ, cần công bố quy tắc phân loại và mức đồng thuận. Annotation có thể bất đồng ở bình luận thiếu ngữ cảnh. Không tìm được adversarial example không chứng minh robustness tuyệt đối; thiết lập chính chỉ xét attacker được xem điểm số.

## Ethics Statement

Annotation có thông báo về nội dung xúc phạm và cơ chế bỏ qua mẫu gây khó chịu. Chia sẻ dữ liệu dẫn xuất tuân theo điều kiện ViHSD; nếu không được phân phối lại văn bản, cung cấp ID, metadata và mã tái dựng cho người có quyền truy cập. Tài nguyên được xây dựng nhằm đánh giá điểm yếu và hỗ trợ cải thiện hệ thống phát hiện nội dung độc hại.

## References

1. Luu et al. 2021. [A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts](https://arxiv.org/abs/2103.11528).
2. Li et al. 2019. [TextBugger: Generating Adversarial Text Against Real-world Applications](https://www.ndss-symposium.org/ndss-paper/textbugger-generating-adversarial-text-against-real-world-applications/). NDSS.
3. Jin et al. 2020. [Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment](https://ojs.aaai.org/index.php/AAAI/article/view/6311). AAAI.
4. Li et al. 2020. [BERT-ATTACK: Adversarial Attack Against BERT Using BERT](https://aclanthology.org/2020.emnlp-main.500/). EMNLP.
5. Ni et al. 2024. [Fraud's Bargain Attack: Generating Adversarial Text Samples via Word Manipulation Process](https://doi.org/10.1109/TKDE.2024.3349708). TKDE.
6. Przybyła et al. 2025. [Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models](https://aclanthology.org/2025.emnlp-main.1405/). EMNLP.
7. Morris et al. 2020. [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://aclanthology.org/2020.emnlp-demos.16/). EMNLP, System Demonstrations.
8. Morris et al. 2020. [Reevaluating Adversarial Examples in Natural Language](https://aclanthology.org/2020.findings-emnlp.341/). Findings of EMNLP.
9. Lent. 2025. [Beyond Weaponization: NLP Security for Medium and Lower-Resourced Languages in Their Own Right](https://arxiv.org/abs/2507.03473). Preprint.
10. Nguyen and Nguyen. 2020. [PhoBERT: Pre-trained Language Models for Vietnamese](https://aclanthology.org/2020.findings-emnlp.92/). Findings of EMNLP.
11. Nguyen et al. 2023. [ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing](https://aclanthology.org/2023.emnlp-main.315/). EMNLP.
12. Conneau et al. 2020. [Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747/). ACL.
13. Xu et al. 2024. [An LLM can Fool Itself: A Prompt-Based Adversarial Attack](https://openreview.net/pdf?id=VVgGbB9TNV). ICLR. [Mã nguồn PromptAttack](https://github.com/GodXuxilie/PromptAttack).

14. Fang et al. 2025. [Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors](https://aclanthology.org/2025.emnlp-main.433/). EMNLP.
15. Qwen Team. 2025. [Qwen3-8B model card và hướng dẫn non-thinking](https://huggingface.co/Qwen/Qwen3-8B); [mô tả hỗ trợ đa ngôn ngữ Qwen3](https://qwenlm.github.io/blog/qwen3/). Tài liệu model chính thức.
