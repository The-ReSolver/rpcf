@testset "Snapshot Time Error    " begin
    e1 = DNSAnalyse.SnapshotTimeError(5.235)
    e2 = DNSAnalyse.SnapshotTimeError(2.14, 8.134) 

    @test e1 isa DNSAnalyse.SnapshotTimeError{1, Float64}
    @test e2 isa DNSAnalyse.SnapshotTimeError{2, Float64}
    @test_throws "Snapshot does not exist at: 5.235" throw(e1)
    @test_throws "Snapshots do not exist between 2.14 - 8.134" throw(e2)
end